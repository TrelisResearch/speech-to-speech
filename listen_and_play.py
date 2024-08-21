import socket
import numpy as np
import threading
from queue import Queue
from dataclasses import dataclass, field
import sounddevice as sd
from transformers import HfArgumentParser
import webrtcvad

@dataclass
class ListenAndPlayArguments:
    send_rate: int = field(
        default=16000,
        metadata={
            "help": "In Hz. Default is 16000."
        }
    )
    recv_rate: int = field(
        default=44100,
        metadata={
            "help": "In Hz. Default is 44100."
        }
    )
    list_play_chunk_size: int = field(
        default=1024,
        metadata={
            "help": "The size of data chunks (in bytes). Default is 1024."
        }
    )
    host: str = field(
        default="localhost",
        metadata={
            "help": "The hostname or IP address for listening and playing. Default is 'localhost'."
        }
    )
    send_port: int = field(
        default=12345,
        metadata={
            "help": "The network port for sending data. Default is 12345."
        }
    )
    recv_port: int = field(
        default=12346,
        metadata={
            "help": "The network port for receiving data. Default is 12346."
        }
    )
    vad_frame_duration: int = field(
        default=30,
        metadata={
            "help": "The VAD frame duration in milliseconds. Default is 30."
        }
    )
    vad_mode: int = field(
        default=3,
        metadata={
            "help": "VAD aggressiveness mode, 0-3. 0 is the least aggressive about filtering out non-speech, 3 is the most aggressive. Default is 3."
        }
    )

def listen_and_play(
    send_rate=16000,
    recv_rate=44100,
    list_play_chunk_size=1024,
    host="localhost",
    send_port=12345,
    recv_port=12346,
    vad_frame_duration=30,
    vad_mode=3,
):
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_socket.connect((host, send_port))

    recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    recv_socket.connect((host, recv_port))

    print("Starting in listening mode...")

    stop_event = threading.Event()
    recv_queue = Queue()
    send_queue = Queue()
    system_speaking = threading.Event()

    vad = webrtcvad.Vad(vad_mode)
    vad_frame_length = int(send_rate * vad_frame_duration / 1000)

    def callback_recv(outdata, frames, time, status):
        if system_speaking.is_set() and not recv_queue.empty():
            data = recv_queue.get()
            outdata[:len(data)] = data
            outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
        else:
            outdata[:] = b'\x00' * len(outdata)

    def callback_send(indata, frames, time, status):
        data = bytes(indata)
        is_speech = vad.is_speech(data, send_rate)
        
        if is_speech and system_speaking.is_set():
            print("User speech detected. Switching to listening mode.")
            system_speaking.clear()
            recv_queue.queue.clear()
        
        if not system_speaking.is_set():
            send_queue.put(data)

    def send(stop_event, send_queue):
        while not stop_event.is_set():
            data = send_queue.get()
            send_socket.sendall(data)

    def recv(stop_event, recv_queue):
        def receive_full_chunk(conn, chunk_size):
            data = b''
            while len(data) < chunk_size:
                packet = conn.recv(chunk_size - len(data))
                if not packet:
                    return None  # Connection has been closed
                data += packet
            return data

        while not stop_event.is_set():
            data = receive_full_chunk(recv_socket, list_play_chunk_size * 2)
            
            if data:
                if not system_speaking.is_set():
                    print("System speech detected. Switching to playing mode.")
                system_speaking.set()
                recv_queue.put(data)

    try:
        send_stream = sd.RawInputStream(samplerate=send_rate, channels=1, dtype='int16', blocksize=vad_frame_length, callback=callback_send)
        recv_stream = sd.RawOutputStream(samplerate=recv_rate, channels=1, dtype='int16', blocksize=list_play_chunk_size, callback=callback_recv)
        threading.Thread(target=send_stream.start).start()
        threading.Thread(target=recv_stream.start).start()

        send_thread = threading.Thread(target=send, args=(stop_event, send_queue))
        send_thread.start()
        recv_thread = threading.Thread(target=recv, args=(stop_event, recv_queue))
        recv_thread.start()
        
        print("Press Ctrl+C to stop...")
        while not stop_event.is_set():
            threading.Event().wait(1)  # Wait for 1 second

    except KeyboardInterrupt:
        print("Finished streaming.")

    finally:
        stop_event.set()
        recv_thread.join()
        send_thread.join()
        send_socket.close()
        recv_socket.close()
        print("Connection closed.")

if __name__ == "__main__":
    parser = HfArgumentParser((ListenAndPlayArguments,))
    listen_and_play_kwargs, = parser.parse_args_into_dataclasses()
    listen_and_play(**vars(listen_and_play_kwargs))
