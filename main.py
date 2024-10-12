import argparse
from noodle.game import SnakeGame
from neural.agent import train
import time
# Constants
WIDTH, HEIGHT = 400, 400
CELL_SIZE = 25
FPS = 120

def main(manual=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", action="store_true", help="Flag to indicate if a human is playing")
    args = parser.parse_args()

    game = SnakeGame(WIDTH, HEIGHT, CELL_SIZE, FPS)
    if args.manual:
        game.play()
    else:
        train(WIDTH, HEIGHT, CELL_SIZE, FPS)


if __name__ == "__main__":
    main()
