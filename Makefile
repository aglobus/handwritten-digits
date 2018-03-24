CXX=g++
CXXFLAGS= -std=c++11 -pedantic -Wno-c++11-extensions 
LDFLAGS=
OUT=a.out

.PHONY: all debug clean

all:
	@$(CXX) $(CXXFLAGS) *.cpp $(LDFLAGS) -o $(OUT)

debug:
	@$(CXX) $(CXXFLAGS) -ggdb *.cpp $(LDFLAGS) -o $(OUT)

clean:
	@rm $(OUT)
