from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image, ImageDraw
import numpy as np
import random
import io
import base64
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Maze generation using Depth-First Search algorithm
def generate_maze(width, height):
    # Initialize the maze with walls
    maze = np.ones((height*2+1, width*2+1), dtype=np.uint8)
    
    # Set the start and end points
    start = (0, 1)
    end = (2*width, 2*height-1)
    
    # Mark the start and end
    maze[start[1]][start[0]] = 0
    maze[end[1]][end[0]] = 0
    
    # Create a stack for DFS
    stack = [(1, 1)]
    visited = set()
    visited.add((1, 1))
    
    # Set the initial cell to be a path
    maze[1][1] = 0
    
    # Directions: right, down, left, up
    directions = [(2, 0), (0, 2), (-2, 0), (0, -2)]
    
    while stack:
        x, y = stack[-1]
        
        # Find unvisited neighbors
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < width*2 and 1 <= ny < height*2 and (nx, ny) not in visited:
                neighbors.append((nx, ny, dx, dy))
        
        if neighbors:
            # Choose a random neighbor
            nx, ny, dx, dy = random.choice(neighbors)
            
            # Remove the wall between current cell and chosen neighbor
            maze[y + dy//2][x + dx//2] = 0
            maze[ny][nx] = 0
            
            # Mark as visited and add to stack
            visited.add((nx, ny))
            stack.append((nx, ny))
        else:
            # Backtrack
            stack.pop()
    
    return maze, start, end

# Solve maze using BFS
def solve_maze(maze, start, end):
    height, width = maze.shape
    queue = [start]
    visited = {start: None}
    
    # Directions: right, down, left, up
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    while queue:
        x, y = queue.pop(0)
        
        if (x, y) == end:
            break
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0 and (nx, ny) not in visited:
                queue.append((nx, ny))
                visited[(nx, ny)] = (x, y)
    
    # Reconstruct path
    if end not in visited:
        return None
    
    path = []
    current = end
    while current:
        path.append(current)
        current = visited[current]
    
    return path[::-1]

# Convert maze to image
def maze_to_image(maze, path=None, cell_size=10):
    height, width = maze.shape
    img = Image.new('RGB', (width * cell_size, height * cell_size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw maze
    for y in range(height):
        for x in range(width):
            if maze[y][x] == 1:  # Wall
                draw.rectangle(
                    (x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size),
                    fill='black'
                )
    
    # Draw path if provided
    if path:
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            draw.line(
                (
                    x1 * cell_size + cell_size // 2, 
                    y1 * cell_size + cell_size // 2,
                    x2 * cell_size + cell_size // 2, 
                    y2 * cell_size + cell_size // 2
                ),
                fill='red', width=cell_size // 2
            )
    
    return img

# Convert image to matrix
def image_to_matrix(image_data):
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('L')  # Convert to grayscale
    width, height = img.size
    
    matrix = []
    for y in range(height):
        row = []
        for x in range(width):
            pixel = img.getpixel((x, y))
            # 1 for walls (black), 0 for paths (white)
            row.append(1 if pixel < 128 else 0)
        matrix.append(row)
    
    return matrix

# Save matrix to file
def save_matrix_as_txt(matrix, filename):
    with open(filename, 'w') as f:
        f.write("laberinto = [\n")
        for row in matrix:
            f.write("    " + str(row) + ",\n")
        f.write("]\n")
    return filename

# Image to base64
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_maze', methods=['POST'])
def generate_maze_route():
    width = int(request.form.get('width', 15))
    height = int(request.form.get('height', 15))
    
    # Generate maze
    maze_array, start, end = generate_maze(width, height)
    
    # Convert to image
    maze_img = maze_to_image(maze_array)
    
    # Solve maze
    path = solve_maze(maze_array, start, end)
    solved_img = maze_to_image(maze_array, path)
    
    # Convert images to base64 for display
    maze_base64 = image_to_base64(maze_img)
    solved_base64 = image_to_base64(solved_img)
    
    # Save the matrix as text
    matrix_file = save_matrix_as_txt(maze_array.tolist(), os.path.join(app.config['UPLOAD_FOLDER'], 'maze_matrix.txt'))
    
    # Save images for download
    maze_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'maze.png'))
    solved_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'maze_solved.png'))
    
    return jsonify({
        'maze_img': maze_base64,
        'solved_img': solved_base64,
        'matrix': maze_array.tolist()
    })

@app.route('/image_to_matrix', methods=['POST'])
def image_to_matrix_route():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    # If user doesn't select file, browser also
    # submits an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Process the image
    file_data = file.read()
    matrix = image_to_matrix(file_data)         
    
    # Save matrix for download
    matrix_file = save_matrix_as_txt(matrix, os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_matrix.txt'))
    
    return jsonify({
        'matrix': matrix,
        'matrix_file': 'uploaded_matrix.txt'
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)