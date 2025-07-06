
CXX = g++
CXXFLAGS = -std=c++23 -Wall -Wextra -g
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

OBJDIR = obj
BINDIR = bin

TARGET = $(BINDIR)/vulkan_app
SHADER_SRC = shader/shader.slang
SHADER_SPV = shader/shader.spv

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:%.cpp=$(OBJDIR)/%.o)

$(shell mkdir -p $(OBJDIR) $(BINDIR))

all: $(TARGET)

$(TARGET): $(OBJECTS) $(SHADER_SPV)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Сборка завершена: $(TARGET)"

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SHADER_SPV): $(SHADER_SRC)
	@echo "Компилируем шейдер: $< -> $@"
	slangc $< -target spirv -o $@

%.spv: %.slang
	@echo "Компилируем шейдер: $< -> $@"
	slangc $< -target spirv -stage vertex -entry vertMain -o $@.vert.spv
	slangc $< -target spirv -stage fragment -entry fragMain -o $@.frag.spv
	@echo "Созданы файлы: $@.vert.spv и $@.frag.spv"

clean:
	rm -rf $(OBJDIR) $(BINDIR)
	rm -f *.spv
	@echo "Очистка завершена"

rebuild: clean all

shader: $(SHADER_SPV)

run: $(TARGET)
	./$(TARGET)

debug: CXXFLAGS += -DDEBUG -g
debug: $(TARGET)

release: CXXFLAGS += -O2 -DNDEBUG
release: $(TARGET)

.PHONY: all clean rebuild shaders run debug release check help

$(OBJECTS): $(wildcard *.h) $(wildcard *.hpp)