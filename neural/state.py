"""State module"""
# from noodle.game import SnakeGame
import numpy as np

from noodle.entities import Direction, Point


def get_state(game) -> np.ndarray:
    """Returns the state of the snake."""
    print("Getting state...")
    state = []
    state.extend(get_direction_state(game))
    state.extend(get_danger_state(game))
    state.extend(get_apple_state(game))
    # state.append(game.turns_since_eat)
    state.append(get_apple_angle_state(game))

    return np.array(state, dtype=int)


def get_state_size(game) -> int:
    """Returns the size of the state."""
    return len(get_state(game))


def get_direction_state(game) -> tuple[bool, bool, bool, bool]:
    """Returns the direction state of the snake."""
    snake_direction = game.snake.direction()
    direction_l = snake_direction == Direction.LEFT
    direction_r = snake_direction == Direction.RIGHT
    direction_u = snake_direction == Direction.UP
    direction_d = snake_direction == Direction.DOWN

    print(
        f"\tdirection_l: {direction_l}, direction_r: {direction_r}, direction_u: {direction_u}, direction_d: {direction_d}"
    )
    return direction_l, direction_r, direction_u, direction_d


def get_danger_state(game) -> tuple[bool, bool, bool]:
    """Returns the danger state of the snake."""
    snake_head = game.snake.head()
    snake_size = game.snake.size()

    position_l = Point(snake_head.x, snake_head.y) - Point(snake_size, 0)
    position_r = Point(snake_head.x, snake_head.y) + Point(snake_size, 0)
    position_u = Point(snake_head.x, snake_head.y) - Point(0, snake_size)
    position_d = Point(snake_head.x, snake_head.y) + Point(0, snake_size)

    snake_direction = game.snake.direction()
    direction_l = snake_direction == Direction.LEFT
    direction_r = snake_direction == Direction.RIGHT
    direction_u = snake_direction == Direction.UP
    direction_d = snake_direction == Direction.DOWN

    danger_straight = (
        (direction_r and game.check_collision(position_r))
        or (direction_l and game.check_collision(position_l))
        or (direction_u and game.check_collision(position_u))
        or (direction_d and game.check_collision(position_d))
    )

    danger_right = (
        (direction_u and game.check_collision(position_r))
        or (direction_d and game.check_collision(position_l))
        or (direction_l and game.check_collision(position_u))
        or (direction_r and game.check_collision(position_d))
    )

    danger_left = (
        (direction_d and game.check_collision(position_r))
        or (direction_u and game.check_collision(position_l))
        or (direction_r and game.check_collision(position_u))
        or (direction_l and game.check_collision(position_d))
    )

    print(
        f"\tdanger_straight: {danger_straight}, danger_right: {danger_right}, danger_left: {danger_left}"
    )
    return danger_straight, danger_right, danger_left


def get_apple_state(game) -> tuple[bool, bool, bool, bool]:
    """Returns the apple state of the snake."""
    snake_head = game.snake.head()
    fruit_position = game.fruit.position()

    fruit_l = snake_head.x > fruit_position.x
    fruit_r = snake_head.x < fruit_position.x
    fruit_u = snake_head.y > fruit_position.y
    fruit_d = snake_head.y < fruit_position.y

    print(
        f"\tfruit_l: {fruit_l}, fruit_r: {fruit_r}, fruit_u: {fruit_u}, fruit_d: {fruit_d}"
    )
    return fruit_l, fruit_r, fruit_u, fruit_d


def get_apple_angle_state(game):
    """Returns the normalized angle of the apple in relation to the snake's head."""
    snake_head = game.snake.head()
    fruit_position = game.fruit.position()

    angle = np.arctan2(
        snake_head.y - fruit_position.y, snake_head.x - fruit_position.x
    )
    angle = np.rad2deg(angle)
    angle = (angle + 360) % 360

    normalized_angle = angle / 360.0

    print(f"\tnormalized_angle: {normalized_angle}")
    return normalized_angle
