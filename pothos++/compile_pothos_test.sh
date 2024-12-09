# compile test

# mac
echo "Compiling pothos++ test"
g++ src/pothos_test.cpp src/legendre.cpp src/verho.cpp src/utils.cpp -std=c++17 -o pothos_test -Og

echo "Compiling pothos++"
g++ src/pothos.cpp src/legendre.cpp src/verho.cpp src/utils.cpp -std=c++17 -o pothos -Og

echo "Running pothos++ test"
./pothos_test

echo "Running pothos++ commandline test : legendre 2"
./pothos legendre -f test/small.dump -o small_l -k 2 -p 2

echo "Running pothos++ commandline test : verho"
./pothos verho -f test/small.dump -o small_v -k 2 -p 2
