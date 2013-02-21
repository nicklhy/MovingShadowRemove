CXXFLAGS := `pkg-config --cflags opencv`
LIBS := `pkg-config --libs opencv`
CXXFLAGS += -g

all:app MCSS.a MCSS.so

app:main.cpp MCSS.a
	g++ $(CXXFLAGS) $(LIBS) main.cpp MCSS.a -o app

MCSS.a:MCSS.o
	ar -rc MCSS.a MCSS.o

MCSS.so:MCSS.o
	g++ -shared MCSS.o -o MCSS.so

MCSS.o:MCSS.cpp
	g++ $(CXXFLAGS) -fPIC -c MCSS.cpp -o MCSS.o

clean:
	rm -rf *.o *.so *.a
	rm -f app
