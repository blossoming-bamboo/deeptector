

from enum import Enum
import pathlib
# import moviepy.editor as mp

from cv2 import *
from time import *
import numpy as np
from moviepy.editor import *
from deepfakeutils import *


class DeepTector:

    def __init__(self):

        print('-'*125)
        print('Welcome to "Deeptector"! Start working!')   
        self.face_extractor = face_extractor 
    

    class Models(Enum):

        RESNEXT = 'ResNext'
        XCEPTION = 'Xception'


    class DetectionResult(Enum):
      
        FAKE = 'FAKE!'
        REAL = 'REAL!'

   
    class SampleFormatter:    

        def get_maximal_side(self, height, width) -> int:
            '''Определяет наибольшее значние среди параметров высоты и ширины'''
            
            maximum = max(height, width)
            return maximum
        

        def change_image_size(self, image, image_size):
            '''Меняет размер образца на необходимый для анализа'''
            
            height, width = image.shape[:2]
        
            if width > height:
                height = height * image_size // width
                width = image_size
            else:
                width = width * image_size // height
                height = image_size

            return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


        def make_square_image(self, image):
            '''Форматирует изображение'''
            
            height, width = image.shape[:2]
            maximum = self.get_maximal_side(height, width)
            
            top_side = 0
            bottom_side = maximum - height
            left_side = 0
            right_side = maximum - width
            
            return cv2.copyMakeBorder(image, top_side, bottom_side, left_side, right_side, cv2.BORDER_CONSTANT, value=0)
    

    def download(self):
        bf_url = 'https://drive.google.com/uc?id=1FhJrGLBnnRw_nq0p58NUzlN2rZI9BAQa'
        anchors = 'https://drive.google.com/uc?id=1oqK5yz5ZaWjLP9O_eXhvGL9RD6hOyS6d'
        py_torch_CV_url = 'https://drive.google.com/uc?id=18cBhKpkRflQmyXCVjy1DtVLIUnLPo2bO'
        res_next_url = 'https://drive.google.com/uc?id=1siAcM9uTEoLEeqKFxq5h-hjs4NoHS0SU'
        xception_url = 'https://drive.google.com/uc?id=1--68J6Ipny937AFjJ_AKjhFXFnvnGecV'

        pytorchcv          = 'pytorchcv-0.0.55-py2.py3-none-any.whl'
        ResNextModel       = 'resnext.pth'
        XcePtionModel      = 'xception.pth'
        BlazeFaceModel     = 'blazeface.pth'
        AnchorsNpy         = 'anchors.npy'
       
        gdown.download(py_torch_CV_url, pytorchcv, quiet=False)
        gdown.download(res_next_url, ResNextModel, quiet=False)
        gdown.download(xception_url, XcePtionModel, quiet=False)
        gdown.download(bf_url, BlazeFaceModel, quiet=False)
        gdown.download(anchors, AnchorsNpy, quiet=False)
    
    
    def get_facelist_from_video(self, video_path):
        """Запускает процесс обработки отдельного видео."""
        
        directory = os.path.dirname(video_path)
        files = [ os.path.basename(video_path) ]

        return self.face_extractor.process_videos(directory, files, [0])
    

    def get_best_samples(self, frames):
        '''Оставляет лишь кадры с наибольшим качеством'''
        
        for i in range(len(frames)):
            frame_data = frames[i]
            if len(frame_data["faces"]) > 0:
                frame_data["faces"] = frame_data["faces"][:1]
                frame_data["scores"] = frame_data["scores"][:1]

        # return frame_data

    def make_prediction(self, video_path, input_size, model):
        '''Возваращает коэф отклоения относительно отдельно взятых моделей'''
        
        try:
            
            # Рекомендуемые коэффициенты нормализации
            m, s, size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], input_size
            nt = Normalize(m, s)
            
            # Выполняет поиск лиц в кадрах.
            face_data = self.get_facelist_from_video(video_path)
            self.get_best_samples(face_data)
           
            print('-'*125)
            print('Face samples: ')
            print(f'\t{len(face_data)}')
            
            if len(face_data) > 0:

                # Получим массив с заданными размерами, заполненный нулями
                matrix = np.zeros((frames_per_video, size, size, 3), dtype=np.uint8)

                # Если количество найденных семплов больше нуля, то готовим образцы к модели
                counter = [0][0]  
                param = "faces"       
                for data in face_data:
                    for sample in data[param]:
              
                        # Изменяем размер, требуемый для модели.
                        # Сохраняем соотношение сторон, если необходимо - добавляем 0  
                        formatter = self.SampleFormatter()                
                        resized_face = formatter.change_image_size(sample, size)
                        resized_face = formatter.make_square_image(resized_face)
                        
                        if counter < frames_per_video:
                            matrix[counter] = resized_face
                            counter += 1
                        else:
                            raise Exception
                        
                if counter > 0:
                    # преобразуем в тензор
                    tensor = torch.tensor(matrix, device=device).float()

                    # Подготовка данных перед обработкой.
                    order = (0, 3, 1, 2)
                    tensor = tensor.permute(order)
                    
                    tensor_range = range(len(tensor))
                    for i in tensor_range:
                        tensor[i] = nt(tensor[i] / 255.)

                    # Выключается градиентное вычисление (выч градиента)
                    with torch.no_grad():

                        prediction = model(tensor)
                        prediction = torch.sigmoid(prediction.squeeze())
                      
                        if model is self.Models.RESNEXT:
                            print('-'*125)
                            print('Model ResNext prediction: ')
                            print(f'\t{prediction[:counter].mean().item()}')
                       
                        elif model is self.Models.XCEPTION:
                            print('-'*125)
                            print('Model Xception prediction: ')
                            print(f'\t{prediction[:counter].mean().item()}')

                        # Получаем единичный тензор, 
                        # который является средним значением по всем элементам итерируемой структуры, 
                        # который вычислияется относительно входных данных, учитывая метрику
                       
                        print('-'*125)
                        print('Model prediction: ')
                        print(f'\t{prediction[:counter].mean().item()}')
                        return prediction[:counter].mean().item()

        except Exception as e:
            print(f"Warning was raised while video was processed {video_path}: {str(e)}")
        
        return 1 / 2 # 50%


    def make_predictions_set(self, directory, vids, input_size, model):
        '''Возвращает список коэф отклонения относительно отдельно взятой модели'''
        
        prediction_list = list()
        for i in range(len(vids)):
            filename = vids[i]
            try:
                predicion = self.make_prediction(os.path.join(directory, filename), input_size, model)
                prediction_list.append(predicion)
            except Exception as exp:
                print('-'*125)
                print(exp)
                print(f'Something wrong with {vids[i]} skip prediction')
        
        print('-'*125)
        print(prediction_list)
        
        return prediction_list


    def get_detection_result(self, content):
        '''Возвращает результат предугадыавния'''
        
        r1=0.224
        r2=0.6124
        total = r1 + r2
        r11 = r1/total
        r22 = r2/total
        
        threshold = 0.3

        print('-'*125)
        print('Load Xception pre-trained model...')
        modelXception = loadXceptionModel()
        print('\tDone!')

        print('-'*125)
        print('Load ResNext pre-trained model...')
        modelResNeXt= loadResNextModel()
        print('\tDone!')

        pred_xception = self.make_prediction(content, input_size_xception, modelXception)
        pred_resnext  = self.make_prediction(content, input_size_resnext, modelResNeXt)
       
        pred_ensembel = r22*pred_resnext + r11*pred_xception
        # pred_ensembel = (pred_resnext + pred_xception) / 2

        if pred_ensembel > threshold:
            print('-'*125)
            print('Comparing prediction and threshold:')
            print(f'\t{pred_ensembel} > {threshold}')

            print('-'*125)
            print(f'{content} - {self.DetectionResult.FAKE.value}')

            return self.DetectionResult.FAKE.value
        
        else:
            print('-'*125)
            print('Comparing prediction and threshold:')
            print(f'\t{pred_ensembel} < {threshold}')

            print('-'*125)
            print(f'{content} - {self.DetectionResult.REAL.value}')

            return self.DetectionResult.REAL.value


    def get_detection_result_set(self, directory, videos, input_size, model):
        '''Получить результаты произвольного количества'''        
        
        threshold = 0.3
        
        if model is self.Models.RESNEXT:
            l_model = loadResNextModel()
        
        elif model is self.Models.XCEPTION:
            l_model = loadXceptionModel()

        prediction_list = self.make_predictions_set(directory, videos, input_size, l_model)
        results = []
        
        for pred, video in zip(prediction_list, videos):
            if (p := pred if model is self.Models.XCEPTION else pred) > threshold:
                print('-'*125)
                print(f'Comparing prediction ({"ResNeXt" if model is self.Models.RESNEXT else "Xception"}) and threshold:')
                print(f'\t{pred} > {threshold}')

                results.append(f'{video} - {self.DetectionResult.FAKE.value}')
            
            else:
                print('-'*125)
                print(f'Comparing prediction ({"ResNeXt" if model is self.Models.RESNEXT else "Xception"}) and threshold:')
                print(f'\t{pred} < {threshold}')

                results.append(f'{video} - {self.DetectionResult.REAL.value}')

        print('-'*125)
        for res in results:
            print(res)        
        
        return results


    def get_detecion_result_with_video(self, output, video_file_path):
        '''Возвращает результат предугадыавния. По окончанию работы выводит видео-результат'''

        print('-'*125)
        print(f'Input file: ')
        print(f'\t{video_file_path}')
        
        # video = VideoFileClip(video_file_path) 
        # audio = video.audio 
        # audio.write_audiofile('audio.mp3') 

        # Инициализирую VideoCapture для получения данных из видео
        file_r = cv2.VideoCapture(video_file_path)

        file_r = cv2.VideoCapture(video_file_path)
        
        # Получаю параметры видео.
        width = int(file_r.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(file_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print('-'*125)
        print('Parameters:')  
        print(f'\tHeight: {height}')  
        print(f'\tWidth: {width}')  
        
        # Инициализирую VideoWriter для записи конечного видео на диск
        file_w = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'),  # 'M', 'P', '4', 'V' 
                                    file_r.get(cv2.CAP_PROP_FPS), (width, height))

        # Получаю результат детекции (предсказание на основе анализа изменения кадров)
        result = self.get_detection_result(video_file_path)

        # import subprocess

        # command = f"ffmpeg -i {video_file_path} -ab 160k -ac 2 -ar 44100 -vn audio.mp3"
        # subprocess.call(command, shell=True)

        # Цикл активен, пока не закончились кадры
        while file_r.isOpened():

            # Читаю фрейм.
            is_ok, frame_data = file_r.read() 
            
            # Проверка кадра.
            if not is_ok:
                break
            
            # Коррекция цветовой палитры (опционально)
            # image = frame.copy()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Получаю параметры текста
            size = cv2.getTextSize(result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            x_param = int((frame_data.shape[1] - size[0]) / 2)
            y_param = int((frame_data.shape[0] + size[1]) / 2)
            
            # 'Кладу' текст в кадр
            cv2.putText(frame_data, result, (x_param, y_param), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Записываю видео на диск покадрово
            file_w.write(frame_data)

        # Работа VideoCapture и VideoWriter окончена
        file_r.release(), file_w.release()
        
        print('_'*100)
        print(result)

        # audio = mp.AudioFileClip("audio.mp3")
        # video1 = mp.VideoFileClip("res.mp4")
        # final = video1.set_audio(audio)

        # final.write_videofile("res.mp4")

        # Инициализирую VideoCapture для получения данных из видео
        player = cv2.VideoCapture(output)
        
        # Проверка на правильность открытия
        if (player.isOpened()== False):
            print("Error opening video file")
        
        # Чтение до тех пор, пока видео не закончилось
        while(player.isOpened()):
            
        # Capture frame-by-frame
            ok, frame = player.read()
            if ok == True:
            # Отображение кадра
                cv2.imshow('Frame', frame)
                
            # Нажатие клавиши "Q" для выхода
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        
        # Выход из цикла если кадр вычита ошибочно
            else:
                break
        
        # Выход из плеера
        player.release()
        
        # Закрывает окно
        cv2.destroyAllWindows()

        return result


def __main__():

    print('-'*125)
    while True:
        print('Choose what you want:\n1 - Get text result about one video\n2 - Get text result about few video\n3 - Get video result about one video')
        func = input('Mode: ')
        if func in ['1', '2', '3']:
            break
        else:
            print('-'*125)
            print(f'Incorrect arg {func}. Try again')

    detector = DeepTector()
    
    if func == '1':
        videos = []
        video_folder = pathlib.Path('examples')
        for video in video_folder.glob('*'):
            if not video.name.endswith('_tested.mp4'):
                videos.append(video_folder / video.name)

        print('-'*125)
        print('Choose video to analyze: ')        
        num = 1
        num_vid = {}
        for video in videos:
            print(f'\t{num} - {str(video)}')        
            num_vid.update({num: video})
            num += 1
        
        while True:
            video_number = input('Enter the number: ')

            if video_number in [str(x) for x in num_vid]:
                break
            else:
                print('-'*125)
                print(f'Incorrect arg {video_number}. Try again')

        video_path = num_vid[int(video_number)]
        detector.get_detection_result(str(video_path))

        print('-'*125)
        print('Good Bye!')

    elif func == '2':
       
        videos = []
        video_folder = pathlib.Path('examples')
        for video in video_folder.glob('*'):
            if not video.name.endswith('_tested.mp4'):
                videos.append(video_folder / video.name)

        print('-'*125)
        print(f'Choose videos you want to analyze: ')
        
        num = 1
        num_vid = {}
       
        for video in videos:
            print(f'\t{num} - {str(video)}')        
            num_vid.update({num: video})
            num += 1

        selected_count = 0
        selected_vids = []
       
        while True:
            sel_number = input('Choose number (press q to exit): ')
          
            if not sel_number in [str(x) for x in num_vid.keys()] and sel_number != 'q':
                print('Incorrect number')
                continue
            
            if sel_number == 'q':
                break
      
            if not num_vid[int(sel_number)] in selected_vids:

                selected_vids.append(num_vid[int(sel_number)])
            else:
                continue
           
            selected_count += 1
            if selected_count > 10:
            
                print('-'*125)
                print('You recieve maximum. Next step')
                break
        
        detector.get_detection_result_set(video_folder.name, 
        [video.name for video in selected_vids], 64, detector.Models.RESNEXT)
    
        print('-'*125)
        print('Good Bye!')
    
    if func == '3':
        videos = []
        video_folder = pathlib.Path('examples')
        for video in video_folder.glob('*'):
            if not video.name.endswith('_tested.mp4'):
                videos.append(video_folder / video.name)

        print('-'*125)
        print('Choose video to analyze: ')        
        num = 1
        num_vid = {}
        for video in videos:
            print(f'\t{num} - {str(video)}')        
            num_vid.update({num: video})
            num += 1
        
        while True:
            video_number = input('Enter the number: ')

            if video_number in [str(x) for x in num_vid]:
                break
            else:
                print('-'*125)
                print(f'Incorrect arg {video_number}. Try again')

        video_path = num_vid[int(video_number)]
        detector.get_detecion_result_with_video('result.mp4', str(video_path))

        print('-'*125)
        print('Good Bye!')


__main__()
