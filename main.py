import argparse

# from neural.agent import train
from noodle import Controller, Model, View

# Constants
WIDTH, HEIGHT = 400, 400
CELL_SIZE = 25
FPS = 15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Flag to indicate if a human is playing",
    )
    args = parser.parse_args()

    game_model = Model(width=WIDTH, height=HEIGHT, cell_size=CELL_SIZE)
    game_view = View(width=WIDTH, height=HEIGHT, cell_size=CELL_SIZE)

    if args.manual:
        game_controller = Controller(game_model, game_view, FPS)
        game_controller.play()
    else:
        pass
        # train(game_model, WIDTH, HEIGHT, CELL_SIZE, FPS)


if __name__ == "__main__":
    main()
