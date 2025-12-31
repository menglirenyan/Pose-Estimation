import cv2


def find_camera():
    # 尝试索引 0, 1, 2 以及不同的 API 后端
    for index in [1, 0, 2]:
        for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
            api_name = "DSHOW" if api == cv2.CAP_DSHOW else "MSMF" if api == cv2.CAP_MSMF else "AUTO"
            print(f"正在尝试: 索引 {index}, 模式 {api_name}...")

            if api is not None:
                cap = cv2.VideoCapture(index, api)
            else:
                cap = cv2.VideoCapture(index)

            if cap.isOpened():
                # 给硬件一点启动时间
                cv2.waitKey(500)
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"🎉 找到啦！请使用：cap = cv2.VideoCapture({index}, cv2.CAP_DSHOW)")
                    cv2.imshow("Success!", frame)
                    cv2.waitKey(3000)
                    cap.release()
                    cv2.destroyAllWindows()
                    return index, api
                cap.release()
    print("❌ 未发现可用画面，请检查摄像头物理连接或隐私设置。")
    return None, None


find_camera()