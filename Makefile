CC = gcc
LDFLAGS = -lm
CFLAGS = -Wall -Wextra -g

COMMON_SRCS = game.c ai.c
COMMON_OBJS = $(COMMON_SRCS:.c=.o)

TEST_TARGET = test
MAIN_TARGET = main

all: $(TEST_TARGET) $(MAIN_TARGET)

$(TEST_TARGET): test.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(MAIN_TARGET): main.o $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(COMMON_OBJS) test.o main.o $(TEST_TARGET) $(MAIN_TARGET)

.PHONY: all clean
