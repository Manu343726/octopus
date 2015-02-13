/* 
 * File:   vector.hpp
 * Author: manu343726
 *
 * Created on 10 de febrero de 2015, 18:54
 */

#include <array>
#include <boost/operators.hpp>
#include <manu343726/turbo_core/turbo_core.hpp>
#include <manu343726/turbo_computing/float.hpp>

#ifndef VECTOR_HPP
#define	VECTOR_HPP

using namespace tml::placeholders;

namespace math
{
    template<typename T, std::size_t Dim>
    class vector
        : boost::addable< vector<T,Dim>             // vector + vector
    	, boost::subtractable< vector<T,Dim>        // vector - vector
    	, boost::dividable2< vector<T,Dim>, T       // vector / T
    	, boost::multipliable2< vector<T,Dim>, T    // vector * T, T * vector
    	, boost::equality_comparable< vector<T,Dim> // vector != vector
     > > > > >
    {
        using compile_time_zero = tml::conditional<std::is_floating_point<T>,
                                                   tml::floating::integer<0>,
                                                   tml::integer<0>
                                                  >;
    public:
        vector() : 
            _coords( tml::to_runtime<tml::repeat<compile_time_zero, tml::size_t<Dim>>>() )
        {}
        
        template<typename... Cs, 
                 TURBO_ENABLE_FUNCTION_IF(tml::all_of<tml::lambda<_1, std::is_convertible<_1,T>>, tml::list<Cs...>>),
                 TURBO_ENABLE_FUNCTION_IF(tml::boolean<Dim == sizeof...(Cs)>)
                >
        vector(Cs... coords) : 
            _coords{ static_cast<T>(coords)... }
        {}
        
        constexpr std::size_t rank() const
        {
            return Dim;
        }
            
        T operator()(std::size_t i) const
        {
            return _coords[i];
        }
        
        T& operator()(std::size_t i)
        {
            return _coords[i];
        }
        
        T operator[](std::size_t i) const
        {
            return (*this)(i);
        }
        
        T& operator[](std::size_t i)
        {
            return (*this)(i);
        }
        
        template<typename... Ds,
                 TURBO_ENABLE_FUNCTION_IF(tml::all_of<tml::lambda<_1, std::is_convertible<_1, std::size_t>>, tml::list<Ds...>>),
                 TURBO_ENABLE_FUNCTION_IF(tml::boolean<(sizeof...(Ds) <= Dim) && (sizeof...(Ds) > 1)>)
                >
        vector<T,sizeof...(Ds)> operator()(Ds... dimensions) const
        {
            return { (*this)(dimensions)... };
        }
        
        T squared_length() const
        {
            return std::accumulate( std::begin(_coords), std::end(_coords),0, [](T current, T e)
            {
               return current + e*e; 
            });
        }
        
        T length() const
        {
            return std::sqrt(squared_length());
        }
        
        vector& operator+=(const vector& v)
        {
            for(std::size_t i = 0; i < rank(); ++i)
                _coords[i] += v[i];
            
            return *this;
        }
        
        vector& operator-=(const vector& v)
        {
            for(std::size_t i = 0; i < rank(); ++i)
                _coords[i] -= v[i];
            
            return *this;
        }
        
        vector& operator*=(T k)
        {
            for(std::size_t i = 0; i < rank(); ++i)
                _coords[i] *= k;
            
            return *this;
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
    };
    
    template<typename T, std::size_t Dim>
    struct vector_dim_proxy
    {
        vector_dim_proxy(vector<T,Dim>& v, std::size_t i) :
            _v{v},
            _i{i}
        {}
            
        T& operator=(const T& e)
        {
            return _v[_i] = e;
        }
        
        operator T() const
        {
            return _v[_i];
        }
        
        operator T&()
        {
            return _v[_i];
        }
    private:
        vector<T,Dim>& _v;
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
        
        proxy_t x, y, z;
    };
}

#endif	/* VECTOR_HPP */

