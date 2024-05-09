from utils import Player, WINDOW_WIDTH
import cv2
import numpy as np


player = Player()
# Initializing a Player object with a random start position on a randomly generated Maze


def strategy():
    # This function is to localize the position of the newly created player with respect to the map
    pass


if __name__ == "__main__":
    strategy()
    map = np.array(player.getMap())
    cv2.imwrite("map.png", map * 255)

    snap = np.array(player.getSnapShot())
    cv2.imwrite("snapshot.png", snap * 255)

