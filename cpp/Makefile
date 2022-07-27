SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

EXE := $(BIN_DIR)/main
SRC := $(wildcard $(SRC_DIR)/*.cpp)
OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# -I is a preprocessor flag, not a compiler flag
CPPFLAGS := -Iinclude -Ilib/include -MMD -MP
# C++ compiler flag
CXXFLAGS := -std=c++17 -Wall
# Linker parameter flags, -L is a linker flag
LDFLAGS  := -Llib/lib
# Linker library flags
LDLIBS   := -llibrary

show:
	@echo $(SRC)
	@echo $(OBJ)

.PHONY: all clean

# debug
debug: CXXFLAGS += -DDEBUG -O0 -g
debug: all

# release
release: CXXFLAGS += -DRELEASE -O3 -s
release: all

all: $(EXE)

$(EXE): $(OBJ) | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	@$(RM) -rv $(BIN_DIR) $(OBJ_DIR)

-include $(OBJ:.o=.d)