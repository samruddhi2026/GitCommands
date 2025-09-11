import tkinter as tk
from tkinter import messagebox
from datetime import datetime
import cv2
import os
from PIL import Image
import numpy as np
import mysql.connector
import threading

# --- Global Variables ---
cctv_url = ""

# --- Helper Functions ---
def update_time():
    """Update the time label with the current date and time."""
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    time_label.config(text=current_time)
    window.after(1000, update_time)

def face_crop(img, face_classifier):
    """Crops and returns the face from an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        crop_faces = img[y:y+h, x:x+w]
        return crop_faces
    return None

def train_classifier():
    """Trains the face recognition classifier and saves it."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        messagebox.showerror('Error', "Data directory not found. Generate dataset first.")
        return

    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:
        try:
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')
            id = int(os.path.split(image)[1].split(".")[1])
            faces.append(imageNp)
            ids.append(id)
        except Exception as e:
            print(f"Error processing image {image}: {e}")
    
    if len(faces) == 0:
        messagebox.showinfo('Result', 'No faces found to train the classifier.')
        return

    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Result', 'Training Completed!')

def frame_boundary(img, classifier, clf):
    """Draws a boundary around detected faces and displays user info."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, 1.1, 10)

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        try:
            id, pred = clf.predict(gray_img[y:y+h, x:x+w])
            confidence = int(100 * (1 - pred / 300))
            
            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password="#mysqlrp3050",
                database="authentic_users"
            )
            my_cursor = mydb.cursor()
            my_cursor.execute("SELECT name FROM users WHERE id=" + str(id))
            result = my_cursor.fetchone()
            user_name = result[0] if result else "Unknown"

            if confidence > 80:
                cv2.putText(img, user_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "Unknown", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        except Exception as e:
            print(f"Recognition or DB error: {e}")
            cv2.putText(img, "Unknown", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    return img

# --- Main Functions (Button Commands) ---
def start_cctv_feed():
    """Starts the CCTV video feed and begins face recognition."""
    cctv_url = t3.get()
    
    if not cctv_url:
        messagebox.showerror("Error", "Please enter a valid CCTV camera URL.")
        return
    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()

    try:
        clf.read("classifier.xml")
    except cv2.error:
        messagebox.showerror("Error", "Classifier file 'classifier.xml' not found or corrupted. Please train the classifier first.")
        return

    video_capture = cv2.VideoCapture(cctv_url)
    if not video_capture.isOpened():
        messagebox.showerror("Error", f"Cannot open video stream: {cctv_url}")
        return

    while True:
        ret, img = video_capture.read()
        if not ret:
            break
        
        img = frame_boundary(img, face_cascade, clf)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1) == 13: # Press Enter to exit
            break

    video_capture.release()
    cv2.destroyAllWindows()

# The video capture loop is now in a separate function to be called by a thread.
def _generate_dataset():
    """Generates a dataset of a person's face for training (internal function)."""
    if not t1.get() or not t2.get():
        messagebox.showinfo('Result', 'Please complete Info')
        return

    mydb = None
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="#mysqlrp3050",
            database="authentic_users"
        )
        my_cursor = mydb.cursor()
        
        my_cursor.execute("SELECT MAX(id) FROM users")
        last_id = my_cursor.fetchone()[0]
        id = (last_id or 0) + 1
        
        insert_query = "INSERT INTO users(id, name, class) VALUES (%s, %s, %s)"
        val = (id, t1.get(), t2.get())
        my_cursor.execute(insert_query, val)
        mydb.commit()

        folder_path = "data"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cropped_face = face_crop(frame, face_classifier)
            if cropped_face is not None:
                img_id += 1
                face = cv2.resize(cropped_face, (640, 480))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = os.path.join(folder_path, f"user.{id}.{img_id}.jpg")
                
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped face", face)

            if cv2.waitKey(1) == 13 or img_id == 200:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Dataset Generated')

    except mysql.connector.Error as e:
        messagebox.showerror("Database Error", str(e))
    finally:
        if mydb and mydb.is_connected():
            my_cursor.close()
            mydb.close()
            
    # Train the classifier after the dataset is generated
    train_classifier()

def generate_dataset_threaded():
    """Starts the dataset generation in a separate thread."""
    threading.Thread(target=_generate_dataset).start()

def delete_user_by_name_class():
    user_name = t5.get()
    user_class = t6.get()
    
    if not user_name or not user_class:
        messagebox.showerror("Error", "Please enter both Name and Class.")
        return
    
    mydb = None
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="#mysqlrp3050",
            database="authentic_users"
        )
        my_cursor = mydb.cursor()
        
        my_cursor.execute("SELECT id FROM users WHERE name=%s AND class=%s", (user_name, user_class))
        user = my_cursor.fetchone()
        
        if not user:
            messagebox.showerror("Error", f"No user found with Name: {user_name} and Class: {user_class}")
            return
        
        user_id = user[0]
        
        my_cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
        mydb.commit()
        
        dataset_dir = "data"
        removed_files = 0
        if os.path.exists(dataset_dir):
            for file in os.listdir(dataset_dir):
                if file.startswith(f"user.{user_id}."):
                    os.remove(os.path.join(dataset_dir, file))
                    removed_files += 1
        
        messagebox.showinfo(
            "Success",
            f"User '{user_name}' from Class '{user_class}' deleted. Removed {removed_files} face data files."
        )
    
    except mysql.connector.Error as e:
        messagebox.showerror("Database Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        if mydb and mydb.is_connected():
            my_cursor.close()
            mydb.close()

# --- GUI Setup ---
window = tk.Tk()
window.title("Face Recognition System")

# Time Label
time_label = tk.Label(window, font=("Times New Roman", 20))
time_label.grid(column=0, row=2, columnspan=2)

# Textboxes and Labels for User Info
l1 = tk.Label(window, text="Name", font=("Times New Roman", 20))
l1.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2 = tk.Label(window, text="Class", font=("Times New Roman", 20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)

# CCTV URL
l3 = tk.Label(window, text="CCTV Camera URL", font=("Times New Roman", 20))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)

# Buttons
b1 = tk.Button(window, text="Train Classifier", font=("Times New Roman", 20), bg='purple', fg='white', command=train_classifier)
b1.grid(column=0, row=4)

b2 = tk.Button(window, text="Detect Faces", font=("Times New Roman", 20), bg='green', fg='white', command=lambda: start_cctv_feed())
b2.grid(column=1, row=4)

# B3 is now changed to call the new threaded function
b3 = tk.Button(window, text="Generate Dataset", font=("Times New Roman", 20), bg='blue', fg='white', command=generate_dataset_threaded)
b3.grid(column=2, row=4)

# Delete User Section
l5 = tk.Label(window, text="User Name", font=("Times New Roman", 20))
l5.grid(column=0, row=6)
t5 = tk.Entry(window, width=50, bd=5)
t5.grid(column=1, row=6)

l6 = tk.Label(window, text="User Class", font=("Times New Roman", 20))
l6.grid(column=0, row=7)
t6 = tk.Entry(window, width=50, bd=5)
t6.grid(column=1, row=7)

delete_button_name_class = tk.Button(
    window,
    text="Delete User by Name & Class",
    font=("Times New Roman", 20),
    bg='red',
    fg='white',
    command=delete_user_by_name_class
)
delete_button_name_class.grid(column=1, row=8)

window.geometry("900x450")
update_time() # Start the time update loop
window.mainloop()