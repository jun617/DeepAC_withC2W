import socket
import struct
import threading
import queue
import cv2
import numpy as np
import os


class NetworkServer:
    """
    멀티스레드 서버 예시

    사용 방법:
      1) server = NetworkServer(host="0.0.0.0", port=7777, save_img=True, save_data=True)
      2) server.start()  # 내부적으로 클라이언트 연결 및 수신 스레드 시작
      3) while True:
             frame_id, header, img = server.get_latest_frame()  # 최신 프레임 받아오기
             # ... CV 처리 등 수행 ...
             # refined_pose, camera_pose = ... (각각 (pos, rot) 형태, 7 float씩)
             server.send_response_pose(refined_pose, camera_pose)
    """

    def __init__(self, host="0.0.0.0", port=7777, save_img=True, save_data=True):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.receive_thread = None

        # 최신 프레임 관리를 위한 Queue (최신 프레임만 꺼내도록 함)
        self.frames_queue = queue.Queue()
        self.running = False

        # 파일 저장 관련 옵션
        self.save_img = save_img      # 이미지 저장 여부
        self.save_data = save_data    # 포즈 데이터 저장 여부

        # 저장 디렉토리 설정
        self.received_dir = "Received"
        self.frame_dir = os.path.join(self.received_dir, "frame")
        os.makedirs(self.received_dir, exist_ok=True)
        os.makedirs(self.frame_dir, exist_ok=True)

    def start(self):
        """서버 소켓 생성 후 클라이언트 연결을 받고, 수신 스레드를 시작한다."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"[Server] Listening on {self.host}:{self.port}")

        self.client_socket, addr = self.server_socket.accept()
        print(f"[Server] Client connected: {addr}")

        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_loop)
        self.receive_thread.start()
        # self.server_socket.listen(5)  # 여러 연결을 대기할 수 있음
        # print(f"[Server] Listening on {self.host}:{self.port}")
        #
        # while True:
        #     print("[Server] Waiting for a new client connection...")
        #     self.client_socket, addr = self.server_socket.accept()
        #     print(f"[Server] Client connected: {addr}")
        #
        #     self.running = True
        #     self.receive_thread = threading.Thread(target=self._receive_loop)
        #     self.receive_thread.start()
        #
        #     # 현재 클라이언트와의 연결이 종료될 때까지 기다림
        #     self.receive_thread.join()
        #     print("[Server] Client disconnected, restarting accept...")

    def _receive_loop(self):
        """
        클라이언트로부터 계속해서 데이터를 수신하여 frames_queue에 저장하고,
        동시에 파일 저장 옵션에 따라 HMD 포즈, 오브젝트 포즈, 이미지를 파일로 저장한다.
        헤더 구조 예시: 104바이트(커스텀 헤더) + JPEG 이미지 데이터
        (실제 패킷 구조는 CustomizedPacket과 동일하게 구성됨)
        """
        while self.running:
            try:
                header_size = 120
                header_bytes = self._recv_all(header_size)
                if not header_bytes:
                    print("[Server] No header bytes (client disconnected?)")
                    self.running = False
                    break

                # '<12i14f' 형식으로 언팩 (12개의 int, 14개의 float)
                unpacked_header = struct.unpack('<8i4f4i14f', header_bytes)
                # unpacked_header 구성 예시:
                # index 0: label, 1~7: 시간정보, 8~11: fx, fy, cx, cy, 12: width, 13: height, 14: data_size, 15: frame_id,
                # index 16~22: HMD Pose (7 float), index 23~29: Object Pose (7 float)
                label = unpacked_header[0]
                frame_id = unpacked_header[15]
                img_size = unpacked_header[14]  # data_size

                # 이미지 데이터 수신
                img_bytes = self._recv_all(img_size)
                if not img_bytes:
                    print("[Server] No image bytes.")
                    self.running = False
                    break

                # JPEG 데이터를 디코딩하여 np.array(BGR) 이미지로 변환
                np_img = np.frombuffer(img_bytes, np.uint8)
                decoded_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                # frames_queue에 (frame_id, unpacked_header, decoded_img) 저장
                self.frames_queue.put((frame_id, unpacked_header, decoded_img))

                # ----- 파일 저장 작업 (옵션에 따라) -----
                if self.save_data:
                    # HMD Pose: unpacked_header 인덱스 12 ~ 18 (7개 float)
                    hmd_pose = unpacked_header[16:23]
                    # Object Pose: unpacked_header 인덱스 19 ~ 25 (7개 float)
                    object_pose = unpacked_header[23:30]

                    camera_intrinsics = unpacked_header[8:12]

                    # HMD_pose.txt에 저장 (각 줄: frame_id, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w)
                    hmd_line = f"{frame_id}, {hmd_pose[0]}, {hmd_pose[1]}, {hmd_pose[2]}, {hmd_pose[3]}, {hmd_pose[4]}, {hmd_pose[5]}, {hmd_pose[6]}\n"
                    with open(os.path.join(self.received_dir, "HMD_pose.txt"), "a") as f:
                        f.write(hmd_line)

                    # object_pose.txt에 저장 (각 줄: frame_id, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w)
                    object_line = f"{frame_id}, {object_pose[0]}, {object_pose[1]}, {object_pose[2]}, {object_pose[3]}, {object_pose[4]}, {object_pose[5]}, {object_pose[6]}\n"
                    with open(os.path.join(self.received_dir, "object_pose.txt"), "a") as f:
                        f.write(object_line)

                    intrinsics_line = f"{frame_id}, {camera_intrinsics[0]}, {camera_intrinsics[1]}, {camera_intrinsics[2]}, {camera_intrinsics[3]}"
                    with open(os.path.join(self.received_dir, "camera_intrinsics.txt"), "a") as f:
                        f.write(intrinsics_line)

                if self.save_img:
                    # 이미지 파일 저장: "frame_{frame_id}.jpg" 이름으로 저장
                    frame_filename = os.path.join(self.frame_dir, f"frame_{frame_id}.jpg")
                    cv2.imwrite(frame_filename, decoded_img)
                    print(f"[Server] Saved frame {frame_id} as {frame_filename}")

            except Exception as e:
                print(f"[Server] Receive loop error: {e}")
                self.running = False
                break

        print("[Server] Receive thread ended.")

    def _recv_all(self, size):
        """size 바이트를 모두 받을 때까지 블로킹 방식으로 읽는다."""
        if not self.client_socket:
            return None

        buf = bytearray()
        while len(buf) < size:
            chunk = self.client_socket.recv(size - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return buf

    def get_latest_frame(self):
        """
        frames_queue에서 최신 프레임만 반환한다.
        큐가 비어 있으면 (None, None, None)을 리턴.
        """
        if self.frames_queue.empty():
            return None, None, None

        latest_frame = None
        while not self.frames_queue.empty():
            latest_frame = self.frames_queue.get()

        if latest_frame is None:
            return None, None, None

        frame_id, header, img = latest_frame
        return frame_id, header, img

    def send_response_pose(self, refined_pose, camera_pose, lost_state):
        """
        클라이언트에게 refined_pose와 camera_pose만 전송한다.
        전송 프로토콜:
          - 응답 헤더: 12바이트
              • 4바이트: frame_id (int, 예시에서는 9999로 고정)
              • 8바이트: 예약용 (0으로 채움)
          - 응답 payload: 56바이트 (14개의 float: refined_pose 7개 + camera_pose 7개)
            총 12 + 56 = 68바이트를 전송한다.
        """
        if not self.client_socket:
            return

        try:
            # 응답 헤더 구성 (12바이트)
            # 첫 4바이트에 frame_id, 이후 8바이트는 0
            frame_id = 9999  # 예시값; 필요에 따라 변경
            header = bytearray()
            header.extend(frame_id.to_bytes(4, 'little', signed=True))
            header.extend((0).to_bytes(4, 'little', signed=True))
            header.extend((0).to_bytes(4, 'little', signed=True))
            # 응답 payload 구성 (56바이트: 14개의 float)
            # refined_pose와 camera_pose는 각각 (pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w) 형태
            payload = struct.pack('<7f', *refined_pose) + struct.pack('<7f', *camera_pose) + struct.pack('<1f', lost_state)
            # 최종 전송: 헤더(12바이트) + payload(56바이트) = 68바이트
            packet = header + payload
            self.client_socket.send(packet)
            print(f"[Server] Sent response pose (68 bytes)")
        except Exception as e:
            print(f"[Server] send_response_pose error: {e}")

    def stop(self):
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join()
        print("[Server] Stopped.")


# if __name__ == '__main__':
#     # 호스트와 포트는 필요에 맞게 수정
#     server = NetworkServer(host='192.168.0.6', port=7777, save_img=True, save_data=True)
#     server.start()
#     print("[MainThread] Server started, waiting for frames...")
#
#     try:
#         while True:
#             # 최신 프레임을 가져온다.
#             frame_id, header, img = server.get_latest_frame()
#             if frame_id is None:
#                 continue
#
#             # 여기서 원하는 CV 처리 등을 수행.
#             # 예시로 refined_pose와 camera_pose를 임의 값으로 설정:
#             refined_pose = (0, 0, 0, 0, 0, 0, 1)  # pos(0,0,0), rot(0,0,0,1)
#             camera_pose = (1, 1, 1, 0, 0, 0, 1)   # pos(1,1,1), rot(0,0,0,1)
#
#             server.send_response_pose(refined_pose, camera_pose)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         server.stop() hey explain this code to me