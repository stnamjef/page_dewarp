PROJNAME      = pagedewarp
SRCS          = src
PROGS         = $(PROJNAME)
CXX           = g++
CXXFLAGS      = -Wall -std=c++11 `pkg-config --cflags opencv`
LIBS          = `pkg-config --libs opencv` -s
PREFIX        = /usr/local
INSTALL       = install
LN            = ln -fs

.PHONY: all clean install

all: $(PROGS)

clean:
	rm -f $(PROGS)

$(PROGS): $(SRCS)/$(PROGS).cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

install: $(PROGS)
	$(INSTALL) -d $(PREFIX)/bin
	$(INSTALL) -m 0755 $(PROGS) $(PREFIX)/bin/
	$(INSTALL) -d $(PREFIX)/share/doc/$(PROJNAME)
	$(INSTALL) -m 0644 LICENSE README.md doc/* $(PREFIX)/share/doc/$(PROJNAME)
