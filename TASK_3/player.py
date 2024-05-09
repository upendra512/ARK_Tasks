from utils import Player, WINDOW_WIDTH
import cv2
import numpy as np


player = Player()
# Initializing a Player object with a random start position on a randomly generated Maze


def strategy():
     player = Player()
    map_image = np.array(player.getMap()) * 255  # Convert map to grayscale image for visualization

    # Capture the snapshot
    snapshot_image = np.array(player.getSnapShot()) * 255

    # Perform template matching
    result = cv2.matchTemplate(map_image, snapshot_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    best_match_loc = max_loc

    # Display the matched location on the map
    map_with_match = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(map_with_match, best_match_loc, (best_match_loc[0] + WINDOW_WIDTH, best_match_loc[1] + WINDOW_WIDTH), (0, 0, 255), 2)

    # Save the map with the matched location
    cv2.imwrite("map_with_match.png", map_with_match)

    return best_match_loc

if __name__ == "__main__":
    best_match_loc = strategy()
    print("Best match location:", best_match_loc)
    num_moves = 100
    moves = 0


    while moves < num_moves:
        # 0: up, 1: down, 2: left, 3: right
        direction = random.choise(0,1,2,3)
        step_size = 1

        if direction == 0:
            player.move_vertical(-step_size)  # Move up
        elif direction == 1:
            player.move_vertical(step_size)   # Move down
        elif direction == 2:
            player.move_horizontal(-step_size) # Move left
        elif direction == 3:
            player.move_horizontal(step_size)  # Move right

        moves += 1
    # This function is to localize the position of the newly created player with respect to the map
    pass

    # This function is to localize the position of the newly created player with respect to the map
    pass


if __name__ == "__main__":
    strategy()
    map = np.array(player.getMap())
    cv2.imwrite("map.png", map * 255)

    snap = np.array(player.getSnapShot())
    cv2.imwrite("snapshot.png", snap * 255)

