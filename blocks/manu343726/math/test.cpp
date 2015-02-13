
#include <manu343726/turbo_core/turbo_core.hpp>
#include <manu343726/math/vector.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>

int main(int argc, char* argv[])
{
    math::vector<int, 3> v{1, 2, 3}, w;
    
    math::vector2<int> q{1,2};
    
    q.x = 22;
    
    std::cout << q << std::endl;
    
    std::cout << v << std::endl
              << v(1,0) << std::endl
              << boost::lexical_cast<math::vector<int,2>>("(1,2)") << std::endl;
}
