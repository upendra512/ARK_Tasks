# Importing maze and idetify obstacles
import cv2
import numpy as np

# Load the image
image = cv2.imread('maze.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a binary inverse threshold
# This step enhances the black lines on a white background
threshold_value = 200  # Adjust this value based on your image lighting conditions
_, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

# Use Canny edge detection
edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

# Detect lines using the Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=10)

# Draw lines on the image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Finding coordinates of start and end
import cv2 

# function to display the coordinates of 
# of the points clicked on the image 
def click_event(event, x, y, flags, params): 

	# checking for left mouse clicks 
	if event == cv2.EVENT_LBUTTONDOWN: 

		# displaying the coordinates 
		# on the Shell 
		print(x, ' ', y) 

		# displaying the coordinates 
		# on the image window 
		font = cv2.FONT_HERSHEY_SIMPLEX 
		cv2.putText(img, str(x) + ',' +
					str(y), (x,y), font, 
					1, (255, 0, 0), 2) 
		cv2.imshow('image', img) 

	# checking for right mouse clicks	 
	if event==cv2.EVENT_RBUTTONDOWN: 

		# displaying the coordinates 
		# on the Shell 
		print(x, ' ', y) 

		# displaying the coordinates 
		# on the image window 
		font = cv2.FONT_HERSHEY_SIMPLEX 
		b = img[y, x, 0] 
		g = img[y, x, 1] 
		r = img[y, x, 2] 
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r), 
					(x,y), font, 1, 
					(255, 255, 0), 2) 
		cv2.imshow('image', img) 

# driver function 
if __name__=="__main__": 

	# reading the image 
	img = cv2.imread('maze.png', 1) 

	# displaying the image 
	cv2.imshow('image', img) 

	# setting mouse handler for the image 
	# and calling the click_event() function 
	cv2.setMouseCallback('image', click_event) 

	# wait for a key to be pressed to exit 
	cv2.waitKey(0) 

	# close the window 
	cv2.destroyAllWindows() 
  # Implementing PRM algorithm
import cv2
import numpy as np
import random
import networkx as nx

# Load the maze image
maze_image = cv2.imread('maze.png', cv2.IMREAD_GRAYSCALE)

# Threshold the image to binary (black and white)
_, binary_image = cv2.threshold(maze_image, 127, 255, cv2.THRESH_BINARY)

# Generate random nodes within free space
def generate_random_node(binary_image):
    h, w = binary_image.shape
    while True:
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        if binary_image[y, x] == 0:  # Check if it's a free space
            return x, y

# Generate k-nearest neighbors for each node
def generate_neighbors(nodes, k, radius):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
        for other_node in nodes:
            if node != other_node and np.linalg.norm(np.array(node) - np.array(other_node)) <= radius:
                G.add_edge(node, other_node)
    return G

# Check if the edge between two nodes is valid
def is_edge_valid(node1, node2, binary_image):
    x1, y1 = node1
    x2, y2 = node2
    pts = np.linspace([x1, y1], [x2, y2], num=100).astype(int)
    for pt in pts:
        x, y = pt
        if binary_image[y, x] == 0:  # If it's an obstacle
            return False
    return True

# Find a path using A* algorithm
def find_path(graph, start, goal):
    try:
        return nx.astar_path(graph, start, goal)
    except nx.NetworkXNoPath:
        print("No path found.")
        return []

# Draw the path on the maze image
def draw_path(maze_image, path):
    for i in range(len(path) - 1):
        cv2.line(maze_image, path[i], path[i + 1], (0, 255, 0), 2)
    return maze_image

# PRM algorithm
def prm(binary_image, num_nodes, k_nearest, connect_radius):
    nodes = [generate_random_node(binary_image) for _ in range(num_nodes)]
    graph = generate_neighbors(nodes, k_nearest, connect_radius)
    start = nodes[0]
    goal = nodes[-1]
    path = find_path(graph, start, goal)
    return path

# Parameters
num_nodes = 100
k_nearest = 5
connect_radius = 30

# Run PRM algorithm
path = prm(binary_image, num_nodes, k_nearest, connect_radius)

# Draw the path on the maze image
maze_image_with_path = draw_path(maze_image.copy(), path)

# Display the result
cv2.imshow('Maze with Path', maze_image_with_path)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the maze image
maze_image = cv2.imread('maze.png', cv2.IMREAD_GRAYSCALE)

# Threshold the image to binary
_, binary_image = cv2.threshold(maze_image, 127, 255, cv2.THRESH_BINARY)

# Display the original and binary images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(maze_image, cmap='gray')
plt.title('Original Maze Image')
plt.subplot(1, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')
plt.show()
# Manually define the start and goal coordinates (x, y)
start_coords = (39,325)  # Adjust as per your maze
goal_coords = (99,337)  # Adjust as per your maze

# Display the start and goal coordinates on the binary image
plt.imshow(binary_image, cmap='gray')
plt.scatter(start_coords[0], start_coords[1], c='red', marker='o', label='Start')
plt.scatter(goal_coords[0], goal_coords[1], c='blue', marker='o', label='Goal')
plt.legend()
plt.title('Start and Goal Coordinates')
plt.show()
import random

# Function to generate random nodes within the free space of the maze
def generate_random_node(binary_image):
    h, w = binary_image.shape
    while True:
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        if binary_image[y, x] == 255:  # Check if it's free space
            return x, y

# Generate random nodes
num_nodes = 100
nodes = [generate_random_node(binary_image) for _ in range(num_nodes)]

# Display the random nodes on the binary image
plt.imshow(binary_image, cmap='gray')
plt.scatter([node[0] for node in nodes], [node[1] for node in nodes], c='green', marker='o', label='Nodes')
plt.scatter(start_coords[0], start_coords[1], c='red', marker='o', label='Start')
plt.scatter(goal_coords[0], goal_coords[1], c='blue', marker='o', label='Goal')
plt.legend()
plt.title('Random Nodes and Start/Goal Coordinates')
plt.show()
import math
import matplotlib.pyplot as plt

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to check if a line segment intersects with black pixels in the maze image
def intersects_obstacle(image, point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy
    while True:
        if image[y1, x1] == 0:  # If the pixel is black (obstacle)
            return True
        if (x1, y1) == (x2, y2):  # If we reached the end point
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return False

# Function to connect neighboring nodes within a specified radius without intersecting obstacles
def connect_nodes(nodes, start_coords, goal_coords, radius, maze_image):
    connected_nodes = []
    lines = []  # List to store the lines connecting nodes
    for node in nodes:
        # Connect the node with start and goal if within radius and without intersecting obstacles
        if (euclidean_distance(node, start_coords) <= radius and
            not intersects_obstacle(maze_image, node, start_coords)):
            connected_nodes.append((node, start_coords))
            lines.append((node, start_coords))
        if (euclidean_distance(node, goal_coords) <= radius and
            not intersects_obstacle(maze_image, node, goal_coords)):
            connected_nodes.append((node, goal_coords))
            lines.append((node, goal_coords))
        # Connect the node with other nodes if within radius and without intersecting obstacles
        for other_node in nodes:
            if node != other_node and euclidean_distance(node, other_node) <= radius:
                if not intersects_obstacle(maze_image, node, other_node):
                    connected_nodes.append((node, other_node))
                    lines.append((node, other_node))
    return connected_nodes, lines

# Define the connection radius
connect_radius = 50  

# Connect neighboring nodes without intersecting obstacles
connected_nodes, lines = connect_nodes(nodes, start_coords, goal_coords, connect_radius, maze_image)

# Display the connected nodes and lines on the binary image
plt.imshow(maze_image, cmap='gray')
plt.scatter([node[0][0] for node in connected_nodes], [node[0][1] for node in connected_nodes], c='cyan', marker='o', label='Connected Nodes')
for line in lines:
    plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], c='magenta')  # Draw lines between connected nodes
plt.scatter(start_coords[0], start_coords[1], c='red', marker='o', label='Start')
plt.scatter(goal_coords[0], goal_coords[1], c='blue', marker='o', label='Goal')
plt.legend()
plt.title('Connected Nodes and Start/Goal Coordinates with Connecting Lines (Avoiding Obstacles)')
plt.show()
