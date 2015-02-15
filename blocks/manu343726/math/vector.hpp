/* 
 * File:   vector.hpp
 * Author: manu343726
 *
 * Created on 10 de febrero de 2015, 18:54
 */

#include <array>
#include <boost/operators.hpp>
#include <boost/functional/overloaded_function.hpp>
#include <manu343726/turbo_core/turbo_core.hpp>
#include <manu343726/turbo_computing/float.hpp>

#ifndef VECTOR_HPP
#define	VECTOR_HPP

using namespace tml::placeholders;

namespace math
{
    namespace detail
    {
        template<typename T>
        struct element_traits
        {
            using value_type = T;
            using reference_type = T&;
        };
        
        template<typename T>
        struct element_traits<std::reference_wrapper<T>>
        {
            using value_type = typename std::remove_reference<decltype(std::declval<std::reference_wrapper<T>>().get())>::type;
            using reference_type = value_type&;
        };
        
        template<typename T>
        struct is_view
        {
            using type = tml::boolean<std::is_same<T, typename element_traits<T>::value_type>::value>;
        };
        
        template<typename F, typename... Fs>
        struct overloaded_function : public F, public overloaded_function<Fs...>
        {
            overloaded_function(F f, Fs... fs) :
                F(f), 
                overloaded_function<Fs...>(fs...)
            {}
        };
        
        template<typename F>
        struct overloaded_function<F> : public F
        {
            overloaded_function(F f) : F(f)
            {}
        };
        
        template<typename... Fs>
        overloaded_function<typename std::decay<Fs>::type...> make_overloaded_function(Fs&&... fs)
        {
            return { fs... };
        }
        
        template<typename... Ts, typename F>
        auto apply_list(tml::list<Ts...>, F f)
        {
            return f(Ts{}...);
        }
        
    }
    
    /**
     * n-dimensional vector template. Supports basic vector algebra such as addition, substraction, etc.
     * 
     * @tparam T: element type. If T is a reference type with value semantics (std::reference_wrapper, pointer, std::shared_ptr)
     *            the vector acts as a vector view (It's elements are really references to another vector/whatever).
     * 
     * @tparam Dim: Rank of the vector (Number of dimensions)
     */
    template<typename T, std::size_t Dim>
    class vector
        : boost::equality_comparable< vector<T,Dim> // vector != vector
     >
    {
        //Ensure compile-time default initialization, don't waste run-time on filling
        //an array of compile-time known size. 
        //NOTE: Turbo's software compile-time floating-point implementation supports
        //double only as runtime representation.
        using compile_time_zero = tml::conditional<std::is_floating_point<T>,
                                                   tml::floating::integer<0>,
                                                   tml::integer<0>
                                                  >;
    public:
        /**
         * Default constructor: Initializes all vector components to zero
         */
        vector() : 
            _coords( tml::to_runtime<tml::repeat<compile_time_zero, tml::size_t<Dim>>>() ) //Note the array of zeroes is generated at compile-time
        {}
        
        /**
         * Initilizes the vector with the specied coordinates
         * @param coords pack of coordinates
         * 
         * Note only values convertible to T can be passed, and the number of coordinates should be equal to vector's rank.
         */
        template<typename... Cs, 
                 TURBO_ENABLE_FUNCTION_IF(tml::all_of<tml::lambda<_1, std::is_convertible<_1,T>>, tml::list<Cs...>>),
                 TURBO_ENABLE_FUNCTION_IF(tml::boolean<Dim == sizeof...(Cs)>)
                >
        vector(Cs... coords) : 
            _coords{ static_cast<T>(coords)... }
        {}
            
    public: 
        template<typename U>
        vector& operator=(const vector<U,Dim>& v)
        {
            for(std::size_t i = 0; i < v.rank(); ++i)
                (*this)(i) = v(i);
            
            return *this;
        }
        
        /**
         * Retuns the number of dimensions of the vector. This function is evaluated at compile-time.
         * @return std::size_t with vector's rank. 
         */
        constexpr std::size_t rank() const
        {
            return Dim;
        }
        
        /**
         * Returns the i-th coordinate of the vector
         * @param i coordinate index (dimension), from 0 to rank() - 1
         * @return The value of the i-th coordinate
         */
        auto operator()(std::size_t i) const
        {
            return _at(i);
        }
        
        /**
         * Returns the i-th coordinate of the vector
         * @param i coordinate index (dimension), from 0 to rank() - 1
         * @return A reference to the i-th coordinate
         */
        auto& operator()(std::size_t i)
        {
            return _at(i);
        }
        
        /**
         * Returns the i-th coordinate of the vector
         * @param i coordinate index (dimension), from 0 to rank() - 1
         * @return The value of the i-th coordinate
         */
        auto operator[](std::size_t i) const
        {
            return (*this)(i);
        }
        
        /**
         * Returns the i-th coordinate of the vector
         * @param i coordinate index (dimension), from 0 to rank() - 1
         * @return A reference to the i-th coordinate
         */
        auto& operator[](std::size_t i)
        {
            return (*this)(i);
        }
        
        /**
         * Returns a vector filled with the specified coordinates from this vector.
         * @param dimensions Set of coordinate indices (dimensions) to fill the vector with.
         * @return a vector of rank sizeof...(dimensions) with the values of the requested coordinates.
         */
        template<typename... Ds,
                 TURBO_ENABLE_FUNCTION_IF(tml::all_of<tml::lambda<_1, std::is_convertible<_1, std::size_t>>, tml::list<Ds...>>),
                 TURBO_ENABLE_FUNCTION_IF(tml::boolean<(sizeof...(Ds) <= Dim) && (sizeof...(Ds) > 1)>)
                >
        vector<typename math::detail::element_traits<T>::value_type,sizeof...(Ds)> operator()(Ds... dimensions) const
        {
            return { (*this)(dimensions)... };
        }
        
        /**
         * Returns a vector filled with references to the specified coordinates from this vector.
         * @param dimensions Set of coordinate indices (dimensions) to fill the vector with.
         * @return a vector of rank sizeof...(dimensions) with references to the requested coordinates.
         * 
         * Note mutating the resulting vector results in mutating this too, since the returned vector acts as
         * a view.
         */
        template<typename... Ds,
                 TURBO_ENABLE_FUNCTION_IF(tml::all_of<tml::lambda<_1, std::is_convertible<_1, std::size_t>>, tml::list<Ds...>>),
                 TURBO_ENABLE_FUNCTION_IF(tml::boolean<(sizeof...(Ds) <= Dim) && (sizeof...(Ds) > 1)>)
                >
        vector<std::reference_wrapper<typename math::detail::element_traits<T>::value_type>,sizeof...(Ds)> operator()(Ds... dimensions)
        {
            return { std::ref((*this)(dimensions))... };
        }
        
        /**
         * Returns a vector filled with references to the specified coordinates from this vector.
         * @param head,tail... Set of coordinate indices (dimensions) to fill the vector with.
         * @return a vector of rank sizeof...(dimensions) with references to the requested coordinates.
         * 
         * Note mutating the resulting vector results in mutating this too, since the returned vector acts as
         * a view.
         */
        template<typename D, typename... Ds>
        auto view(D head, Ds... tail) 
        {
            return (*this)(head, tail...);
        }
        
        /**
         * Returns a view of the whole vector
         */
        auto view()
        {
            using indices = tml::size_t_range<0,Dim>;
            
            auto call = [this](auto... indices)
            {
                return view(indices...);
            };
            
            return math::detail::apply_list(indices{}, call);
        }
        
        /**
         * Returns a vector filled with the specified coordinates from this vector.
         * @param head,tail... Set of coordinate indices (dimensions) to fill the vector with.
         * @return a vector of rank sizeof...(dimensions) with the values of the requested coordinates.
         */
        template<typename D, typename... Ds>
        auto copy(D head, Ds... tail) const
        {   
            return (*this)(head, tail...);
        }
        
        /**
         * Returns a true copy of the vector, even if the vector is a view.
         * If it's a view, returns a copy of the viewed subvector.
         */
        auto copy() const
        {
            using indices_t = tml::size_t_range<0,Dim>;
            
            auto call = [this](auto... indices)
            {
                static_assert(sizeof...(indices) == Dim, "???");
                
                return copy(indices...);
            };
            
            return math::detail::apply_list(indices_t{}, call);
        }
        
        /**
         * Checks if this vector is a view, i.e. references other vector's coordinates or 
         * has its own.
         * @return true if the vector is a view, false otherwise.
         */
        constexpr bool is_view() const
        {
            return tml::eval<math::detail::is_view<T>>::value;
        }
        
        /**
         * Returns the length of the vector, squared.
         */
        T squared_length() const
        {
            return std::accumulate( std::begin(_coords), std::end(_coords),0, [](T current, T e)
            {
               return current + e*e; 
            });
        }
        
        /**
         * Returns the length of the vector.
         */
        T length() const
        {
            return std::sqrt(squared_length());
        }
        
        template<typename U>
        vector& operator+=(const vector<U,Dim>& v)
        {
            for(std::size_t i = 0; i < rank(); ++i)
                _coords[i] += v[i];
            
            return *this;
        }
        
        template<typename U>
        vector& operator-=(const vector<U,Dim>& v)
        {
            for(std::size_t i = 0; i < rank(); ++i)
                _coords[i] -= v[i];
            
            return *this;
        }
        
        vector& operator*=(typename math::detail::element_traits<T>::value_type k)
        {
            for(std::size_t i = 0; i < rank(); ++i)
                _coords[i] *= k;
            
            return *this;
        }
        
        vector& operator/=(typename math::detail::element_traits<T>::value_type k)
        {
            for(std::size_t i = 0; i < rank(); ++i)
                _coords[i] /= k;
            
            return *this;
        }
        
        friend auto operator+(const vector& lhs, const vector& rhs)
        {
            auto tmp = lhs.copy();
            
            return tmp += rhs;
        }
        
        friend auto operator-(const vector& lhs, const vector& rhs)
        {
            auto tmp = lhs.copy();
            
            return tmp -= rhs;
        }
        
        friend auto operator*(const vector& lhs, typename math::detail::element_traits<T>::value_type e)
        {
            auto tmp = lhs.copy();
            
            return tmp *= e;
        }
        
        friend auto operator*(typename math::detail::element_traits<T>::value_type e, const vector& rhs)
        {
            return rhs * e;
        }
        
        friend auto operator/(const vector& lhs, typename math::detail::element_traits<T>::value_type e)
        {
            auto tmp = lhs.copy();
            
            return tmp /= e;
        }
        
        friend bool operator==(const vector& lhs, const vector& rhs)
        {
            for(std::size_t i = 0; i < lhs.rank(); ++i)
            {
                if(lhs(i) != rhs(i))
                    return false;
            }
            
            return true;
        }
        
        friend std::ostream& operator<<(std::ostream& os, const vector& v)
        {
            os << "(";
            
            for(std::size_t i = 0; i < v.rank(); ++i)
            {
                os << v[i];
                
                if(i < v.rank() - 1)
                    os << ",";
            }
            
            return os << ")";
        }
        
        friend std::istream& operator>>(std::istream& is, vector& v)
        {
            char placeholder;
            
            is >> placeholder;
            
            for(std::size_t i = 0; i < v.rank(); ++i)
            {
                is >> v[i];
                
                if(i < v.rank() - 1)
                    is >> placeholder;
            }
            
            return is >> placeholder;
        }
        
    private:
        std::array<T,Dim> _coords;
        
        using value_type = typename math::detail::element_traits<T>::value_type;
        using reference_type = value_type&;
        
        value_type _at(std::size_t i) const
        {
            auto accessor = math::detail::make_overloaded_function(
                [&](auto x) -> decltype(x.get())
                {
                    return x.get();
                },
                [&](value_type x)
                {
                    return x;
                }
            );
            
            return accessor(_coords[i]);
        }
        
        reference_type _at(std::size_t i)
        {
            auto accessor = math::detail::make_overloaded_function(
                [&](auto& x) -> typename std::add_lvalue_reference<decltype(x.get())>::type
                {
                    return x.get();
                },
                [&](reference_type x) -> reference_type
                {
                    return x;
                }
            );
            
            return accessor(_coords[i]);
        }
    };
    
    namespace detail
    {
        template<typename V>
        struct vector_traits
        {};
        
        template<typename T, std::size_t Dim>
        struct vector_traits<vector<T,Dim>>
        {
            using value_type = T;
        };
    }
    
    template<typename T, std::size_t Dim>
    struct vector_dim_proxy
    {
        using value_type = typename math::detail::element_traits<T>::value_type;
        
        vector_dim_proxy(vector<T,Dim>& v, std::size_t i) :
            _v{v},
            _i{i}
        {}
            
        template<typename U>
        value_type& operator=(const U& e)
        {
            return _v.get()[_i] = e;
        }
        
        operator value_type() const
        {
            return _v.get()[_i];
        }
        
        operator value_type&()
        {
            return _v.get()[_i];
        }
    private:
        std::reference_wrapper<vector<T,Dim>> _v;
        std::size_t _i;
    };
    
    template<typename T>
    struct vector1 : public vector<T,1>
    {
        using proxy_t = vector_dim_proxy<T,1>;
        
        template<typename... ARGS>
        vector1(ARGS&&... args) : 
            vector<T,1>{std::forward<ARGS>(args)...},
            x{*this,0}
        {}

        //Allow assignment from views to non-views and vice-versa
        template<typename U>
        vector1& operator=(const vector<U,1>& v)
        {
            vector<T,1>::operator=(v);
            
            return *this;
        }
        
        proxy_t x;
    };
    
    template<typename T>
    struct vector2 : public vector<T,2>
    {
        using proxy_t = vector_dim_proxy<T,2>;
        
        template<typename... ARGS>
        vector2(ARGS&&... args) : 
            vector<T,2>{std::forward<ARGS>(args)...},
            x{*this,0},
            y{*this,1}
        {}
            
        //Allow assignment from views to non-views and vice-versa
        template<typename U>
        vector2& operator=(const vector<U,2>& v)
        {
            vector<T,2>::operator=(v);
            
            return *this;
        }
        
        proxy_t x, y;
    };
    
    template<typename T>
    struct vector3 : public vector<T,3>
    {
        using proxy_t = vector_dim_proxy<T,3>;
        
        template<typename... ARGS>
        vector3(ARGS&&... args) : 
            vector<T,3>{std::forward<ARGS>(args)...},
            x{*this,0},
            y{*this,1},
            z{*this,2}
        {}
            
        //Allow assignment from views to non-views and vice-versa
        template<typename U>
        vector3& operator=(const vector<U,3>& v)
        {
            vector<T,3>::operator=(v);
            
            return *this;
        }
        
        proxy_t x, y, z;
    };
    
    template<typename V>
    vector1<typename math::detail::vector_traits<V>::value_type> vector1_adaptor(V&& v)
    {
        return { v };
    }
    
    template<typename V>
    vector2<typename math::detail::vector_traits<V>::value_type> vector2_adaptor(V&& v)
    {
        return { v };
    }
    
    template<typename V>
    vector3<typename math::detail::vector_traits<V>::value_type> vector3_adaptor(V&& v)
    {
        return { v };
    }
}

#endif	/* VECTOR_HPP */

