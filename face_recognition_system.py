import cv2
import numpy as np
import pandas as pd
import face_recognition
import os
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import logging

class FaceRecognitionSystem:
    def __init__(self, faces_directory: str, csv_file: str, encodings_file: str = "face_encodings.pkl"):
        """
        Initialize the Face Recognition System
        
        Args:
            faces_directory: Directory containing known face images
            csv_file: CSV file with face information (should have 'filename' and other info columns)
            encodings_file: File to save/load face encodings for faster processing
        """
        self.faces_directory = Path(faces_directory)
        self.csv_file = csv_file
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_info_df = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load face information from CSV
        self.load_face_info()
        
        # Load or generate face encodings
        self.load_or_generate_encodings()
    
    def load_face_info(self):
        """Load face information from CSV file"""
        try:
            self.face_info_df = pd.read_csv(self.csv_file)
            self.logger.info(f"Loaded {len(self.face_info_df)} records from CSV")
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {e}")
            raise
    
    def load_or_generate_encodings(self):
        """Load existing encodings or generate new ones"""
        if os.path.exists(self.encodings_file):
            self.logger.info("Loading existing face encodings...")
            self.load_encodings()
        else:
            self.logger.info("Generating new face encodings...")
            self.generate_encodings()
            self.save_encodings()
    
    def generate_encodings(self):
        """Generate face encodings for all images in the faces directory"""
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        for image_path in self.faces_directory.iterdir():
            if image_path.suffix.lower() in supported_formats:
                try:
                    # Load and encode face
                    image = face_recognition.load_image_file(str(image_path))
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        # Use the first face found in the image
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(image_path.stem)
                        self.logger.info(f"Encoded face for {image_path.name}")
                    else:
                        self.logger.warning(f"No face found in {image_path.name}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {image_path.name}: {e}")
        
        self.logger.info(f"Generated {len(self.known_face_encodings)} face encodings")
    
    def save_encodings(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
        self.logger.info(f"Saved encodings to {self.encodings_file}")
    
    def load_encodings(self):
        """Load face encodings from file"""
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
            self.known_face_encodings = data['encodings']
            self.known_face_names = data['names']
            self.logger.info(f"Loaded {len(self.known_face_encodings)} face encodings")
        except Exception as e:
            self.logger.error(f"Error loading encodings: {e}")
            raise
    
    def recognize_face(self, image_path: str, tolerance: float = 0.6) -> List[Dict]:
        """
        Recognize faces in the given image and return match information
        
        Args:
            image_path: Path to the image to analyze
            tolerance: Lower values make recognition more strict
            
        Returns:
            List of dictionaries containing match information
        """
        try:
            # Load the image
            unknown_image = face_recognition.load_image_file(image_path)
            unknown_face_encodings = face_recognition.face_encodings(unknown_image)
            
            if not unknown_face_encodings:
                return [{"error": "No face found in the image"}]
            
            results = []
            
            for i, unknown_face_encoding in enumerate(unknown_face_encodings):
                # Compare with known faces
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, unknown_face_encoding
                )
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, unknown_face_encoding, tolerance=tolerance
                )
                
                # Find the best match
                best_match_index = np.argmin(face_distances)
                match_distance = face_distances[best_match_index]
                
                # Calculate match percentage (inverse of distance, normalized)
                match_percentage = max(0, (1 - match_distance) * 100)
                
                if matches[best_match_index] and match_percentage > 50:  # Minimum threshold
                    matched_name = self.known_face_names[best_match_index]
                    face_info = self.get_face_info(matched_name)
                    
                    result = {
                        "face_number": i + 1,
                        "matched": True,
                        "match_percentage": round(match_percentage, 2),
                        "match_distance": round(match_distance, 4),
                        "matched_name": matched_name,
                        "face_info": face_info
                    }
                else:
                    result = {
                        "face_number": i + 1,
                        "matched": False,
                        "match_percentage": round(match_percentage, 2),
                        "match_distance": round(match_distance, 4),
                        "matched_name": None,
                        "face_info": None
                    }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error recognizing face: {e}")
            return [{"error": str(e)}]
    
    def get_face_info(self, filename: str) -> Optional[Dict]:
        """Get information about a face from the CSV database"""
        # Try to match by filename (with or without extension)
        mask = (self.face_info_df['filename'] == filename) | \
               (self.face_info_df['filename'] == f"{filename}.jpg") | \
               (self.face_info_df['filename'] == f"{filename}.jpeg") | \
               (self.face_info_df['filename'] == f"{filename}.png")
        
        matches = self.face_info_df[mask]
        
        if not matches.empty:
            # Return the first match as a dictionary
            return matches.iloc[0].to_dict()
        else:
            self.logger.warning(f"No information found for {filename}")
            return {"filename": filename, "info": "No additional information available"}
    
    def recognize_from_webcam(self, tolerance: float = 0.6):
        """Real-time face recognition from webcam"""
        video_capture = cv2.VideoCapture(0)
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB
            
            # Find faces in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=tolerance
                )
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                
                name = "Unknown"
                match_percentage = 0
                
                if matches and any(matches):
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        match_percentage = max(0, (1 - face_distances[best_match_index]) * 100)
                
                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                
                label = f"{name} ({match_percentage:.1f}%)" if name != "Unknown" else "Unknown"
                cv2.putText(frame, label, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
    
    def add_new_face(self, image_path: str, face_info: Dict):
        """Add a new face to the database"""
        try:
            # Generate encoding for new face
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                return {"error": "No face found in the image"}
            
            # Add to known faces
            filename = Path(image_path).stem
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(filename)
            
            # Add to CSV data
            face_info['filename'] = filename
            self.face_info_df = pd.concat([self.face_info_df, pd.DataFrame([face_info])], 
                                        ignore_index=True)
            
            # Save updated data
            self.face_info_df.to_csv(self.csv_file, index=False)
            self.save_encodings()
            
            self.logger.info(f"Added new face: {filename}")
            return {"success": f"Added new face: {filename}"}
            
        except Exception as e:
            self.logger.error(f"Error adding new face: {e}")
            return {"error": str(e)}


# Example usage and testing
def main():
    # Initialize the system
    face_system = FaceRecognitionSystem(
        faces_directory="known_faces",  # Directory with known face images
        csv_file="face_database.csv"    # CSV file with face information
    )
    
    # Test with all images in the test_images folder
    test_images_dir = Path("test_images")
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    print("\nProcessing test images...")
    for image_path in test_images_dir.iterdir():
        if image_path.suffix.lower() in supported_formats:
            print(f"\nTesting image: {image_path.name}")
            results = face_system.recognize_face(str(image_path))
            
            for result in results:
                if "error" in result:
                    print(f"Error: {result['error']}")
                elif result["matched"]:
                    print(f"Face {result['face_number']} matched:")
                    print(f"  Name: {result['matched_name']}")
                    print(f"  Match Percentage: {result['match_percentage']}%")
                    print(f"  Face Info: {result['face_info']}")
                else:
                    print(f"Face {result['face_number']} - No match found")
                    print(f"  Best match percentage: {result['match_percentage']}%")
    
    # Start webcam recognition (uncomment to use)
    # face_system.recognize_from_webcam()


if __name__ == "__main__":
    main()