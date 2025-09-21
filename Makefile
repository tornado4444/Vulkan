CXX = g++
CXXFLAGS = -std=c++23 -Wall -Wextra -g
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

OBJDIR = obj
BINDIR = bin
SHADERDIR = shaders
TARGET = $(BINDIR)/vulkan_app

# Исходный файл шейдера
SHADER_SRC = shader/shader.slang
# Выходные файлы шейдеров (соответствуют пути в коде)
SHADER_SPV = $(SHADERDIR)/slang.spv

SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:%.cpp=$(OBJDIR)/%.o)

# Создаем необходимые директории
$(shell mkdir -p $(OBJDIR) $(BINDIR) $(SHADERDIR))

all: $(TARGET)

$(TARGET): $(OBJECTS) $(SHADER_SPV)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Сборка завершена: $(TARGET)"

$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Компиляция всех типов шейдеров в один SPIR-V файл
$(SHADER_SPV): $(SHADER_SRC)
	@echo "Компилируем мульти-стейдж шейдер: $< -> $@"
	slangc $< -target spirv -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR) $(SHADERDIR)
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

.PHONY: all clean rebuild shader run debug release

$(OBJECTS): $(wildcard *.h) $(wildcard *.hpp)