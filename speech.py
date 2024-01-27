import speech_recognition as sr
import pyperclip
import pyaudio

def stt(audio):
    r=sr.Recognizer()

    with sr.AudioFile(audio) as source:
        audiodata=r.record(source)

    try:
        text =r.recognize_google(audiodata)
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand"
    except sr.RequestError as e:
        return f"Could not request results from google speech"
def ra():
    r=sr.Recognizer()

    with sr.Microphone() as source:
        print("say something")
        audiodata=r.listen(source)
    try:
        text=r.recognize_google(audiodata)
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand"
    except sr.RequestError as e:
        return f"Could not request results from google speech"

    
print("1.Record audio")
print("pre recorded")
choice=input("Enter ur choice")
        
if choice == '1':
    converted_text = ra()
elif choice == '2':
    file_path = input("Enter the path to the pre-recorded audio file: ")
    converted_text = stt(file_path)
else:
    print("Invalid choice. Exiting.")


print("\nConverted Text:")
print(converted_text)

# Copy the converted text to the clipboard
pyperclip.copy(converted_text)
print("\nText copied to clipboard.")

