"""Main Module."""
from noodle.game import SnakeGame

# Constants
WIDTH, HEIGHT = 500, 500
CELL_SIZE = 25
FPS = 10

def main():
    game = SnakeGame(WIDTH, HEIGHT, CELL_SIZE, FPS)
    game.run()


if __name__ == "__main__":
    main()