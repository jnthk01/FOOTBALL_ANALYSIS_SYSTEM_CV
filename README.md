# FOOTBALL_ANALYSIS_SYSTEM_CV

This project aims to analyze football game footage by tracking players and the ball, estimating speed and distance covered, and transforming player positions from pixel coordinates to real-world measurements.

### Key Features
- **Perspective Transformation**: Transforms the player positions from pixel coordinates to a virtual court view using perspective transformation techniques.
- **Speed & Distance Estimation**: Calculates the speed (in km/h) and distance (in km) covered by players between frames based on their transformed positions.
- **Player Tracking**: Keeps track of player positions throughout multiple frames in a video and calculates their speed and distance over time.
- **Overlay Information**: Displays calculated speed and distance on the video frames to visually track player movement.

## Project Structure
- `utils/bbox_utils.py`: Utility functions for bounding box operations, such as measuring distances and getting foot positions.
- `ViewTransformer`: Class to handle perspective transformations of player positions.
- `SpeedDistanceEstimator`: Class to calculate speed and distance and annotate video frames with these values.
- `main.py`: The main script to run the analysis.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/football-analysis.git
   cd football-analysis
   ```

2. **Create a Virtual Environment**
    ```bash
    python -m venv myenv
    ```

3. **Activate the Virtual Environment**
   - On Windows:
     ```bash
     myenv/Scripts/activate
     ```
   - On macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```

4. **Install the Requirements**
   Install the necessary packages listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the Main Script**
    Once the environment is set up and the dependencies are installed, run the `main.py` script to start the football analysis.
    ```bash
    python main.py
    ```