# Analysing Social Media Images and Text for Disaster Response

## Problem Statement
Disaster response teams require timely and accurate information to prioritize resources and save lives. Social media platforms like Twitter and Instagram provide valuable real-time data during disasters. However, manually processing and filtering large volumes of multimedia posts to extract critical disaster-related content is inefficient. This project aims to develop a system that automatically processes both images and text from social media posts to detect disaster-related information, enabling faster response times and resource allocation.

## Problem Deliverables
The project delivers a comprehensive system capable of:
1. Detecting disaster-related objects and events in images.
2. Analyzing text posts to identify disaster-related information, such as calls for help or damage reports.
3. Integrating image and text classification models into a web application for real-time social media post processing.
4. Ensuring the system can handle multiple requests simultaneously, providing seamless user interaction and scalability.

## Documentation
Link - [Project Documentation](<https://drive.google.com/file/d/1z0sbAYarW81ptaTxwbUjGdRxKHrQuVe3/view?usp=sharing>)

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd Analysing-Social-Media-Images-and-Text-for-Disaster-Response
   ```

2. **Download Model Weights**
   - Download the ResNet model weights from [Google Drive link](<https://drive.google.com/file/d/1-tLyuFq9223xgehtViXDCuc8h6_kB62E/view?usp=sharing>).
   - Place the downloaded file in the `models` folder.

3. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Flask Server**
   ```bash
   flask run
   ```

6. **Access the Web Application**
   - Open a web browser and go to `http://127.0.0.1:5000` to interact with the application.
   - Project HomePage:
   ![image](https://github.com/user-attachments/assets/7840087c-16a8-4ebc-a7f1-377577d3cd6e)


## Usage
- Upload social media images and text posts via the web interface.
- The system will analyze the content and provide disaster-related information.
- Monitor the results and adjust the system settings as needed.

## Contributing
- Fork the repository and make your changes.
- Submit a pull request with a detailed description of your modifications.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, please contact [Rakshit Garg](mailto:rakshitonwork@gmail.com).


