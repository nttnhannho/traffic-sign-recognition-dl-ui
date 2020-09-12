from tkinter import StringVar, Tk, Label, Button, Entry, CENTER, filedialog, END, messagebox, RAISED
import constant as cons
import util
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

class gui_obj(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.title(cons.TITLE)
        self.minsize(width=cons.WIDTH, height=cons.HEIGHT)
        self.resizable(False, False)
        self.iconbitmap(cons.ICON_PATH)
        self.configure(background='darkslategray')
        
        self.input_file = StringVar()
        self.__input_path = ''
        self.model_file = StringVar()
        self.__model_path = ''
        
        self.__is_test_set_img = False
        
        self.lb_topic = Label(self, text=cons.TOPIC, bg='darkslategray', fg="gold", font=("", 23, 'bold'))
        self.lb_topic.place(relx=0.5, rely=0.15, anchor=CENTER)
        
        self.lb_img_info = Label(self, text="", bg='darkslategray', fg="lawngreen", font=("", 10, ''))
        self.lb_img_info.place(relx=0.20, rely=0.31)
        self.lb_input_file = Label(self, text="Input image:", bg='darkslategray', fg="white", font=("", 14, ''))
        self.lb_input_file.place(relx=0.01, rely=0.38)
        self.en_input_file = Entry(self, borderwidth=2, state='readonly', textvariable=self.input_file)
        self.en_input_file.place(relx=0.20, rely=0.38, relwidth=0.62, relheight=0.11)
        self.btn_choose_input_file = Button(self, text ="Choose file", bg='lightgrey', fg="black", borderwidth=4, relief=RAISED, command=lambda:self.upload_image())
        self.btn_choose_input_file.place(relx=0.84, rely=0.38, relwidth=0.14, relheight=0.11)
        
        self.lb_model_file = Label(self, text="Input model:", bg='darkslategray', fg="white", font=("", 14, ''))
        self.lb_model_file.place(relx=0.01, rely=0.55)
        self.en_model_file = Entry(self, borderwidth=2, state='readonly', textvariable=self.model_file)
        self.en_model_file.place(relx=0.20, rely=0.55, relwidth=0.62, relheight=0.11)
        self.btn_choose_model_file = Button(self, text ="Choose file", bg='lightgrey', fg="black", borderwidth=4, relief=RAISED, command=lambda:self.upload_model())
        self.btn_choose_model_file.place(relx=0.84, rely=0.55, relwidth=0.14, relheight=0.11)
        
        self.btn_predict = Button(self, text ="Predict", bg='teal', fg="white", font=("", 14, 'bold'), borderwidth=4, relief=RAISED, command=lambda:self.predict())
        self.btn_predict.place(relx=0.06, rely=0.75, relwidth=0.28, relheight=0.2)
        
        self.btn_clear = Button(self, text ="Clear all", bg='teal', fg="white", font=("", 14, 'bold'), borderwidth=4, relief=RAISED, command=lambda:self.clear_all())
        self.btn_clear.place(relx=0.36, rely=0.75, relwidth=0.28, relheight=0.2)
        
        self.btn_quit = Button(self, text ="Exit", bg='teal', fg="white", font=("", 14, 'bold'), borderwidth=4, relief=RAISED, command=lambda:self.exit_app())
        self.btn_quit.place(relx=0.66, rely=0.75, relwidth=0.28, relheight=0.2)
        
        util.make_center(self)
        
    def __check_img(self):
        if cons.IMG_TESTSET in self.en_input_file.get():
            self.lb_img_info.config(text=cons.IMG_INFO)
            self.__is_test_set_img = True
        else:
            self.lb_img_info.config(text='')
            self.__is_test_set_img = False
        
    def upload_image(self):
        try:
            self.__input_path = filedialog.askopenfilename(title="Select image file", 
                                                         filetypes=[("image files", "*.png *.PNG *.jpg *.JPG *.jpeg *.JPEG")], 
                                                         initialdir='./')
            if self.__input_path:
                self.input_file.set(self.__input_path)
                
            self.__check_img()
        except Exception as ex:
            print(ex)
            messagebox.showerror("Upload Image Error", "Error occurred while uploading image!")
            
    def upload_model(self):
        try:
            self.__model_path = filedialog.askopenfilename(title="Select model file", 
                                                         filetypes=[("model files", "*.h5")], 
                                                         initialdir='./')
            if self.__model_path:
                self.model_file.set(self.__model_path)
        except Exception as ex:
            print(ex)
            messagebox.showerror("Upload Model Error", "Error occurred while uploading model!")

    def predict(self):
        try:
            if not util.check_valid_path(self.en_input_file.get()):
                messagebox.showwarning("Warning","Your input image is invalid.")
            elif not util.check_valid_path(self.en_model_file.get()):
                messagebox.showwarning("Warning","Your input model is invalid.")
            elif util.check_valid_path(self.en_input_file.get()) and util.check_valid_path(self.en_model_file.get()):
                # load label
                print("[INFO]: LOADING LABEL...")
                self.__label_data = pd.read_csv(cons.SIGNNAMES)
                self.__label_values = self.__label_data['SignName'].values
                print("[INFO]: FINISH LOADING LABEL.")
    
                # load Model
                print("[INFO]: LOADING MODEL...")
                self.__model = load_model(self.en_model_file.get())
                print("[INFO]: FINISH LOADING MODEL.")
                
                # PREDICT PROCESS
                print("[INFO]: PREDICTING IMAGE...")
                img = cv2.imread(self.en_input_file.get())
                
                # get ROI values in Test.csv
                y_test = pd.read_csv("./input/Test.csv")
                x1_val = y_test['Roi.X1'].values
                y1_val = y_test['Roi.Y1'].values
                x2_val = y_test['Roi.X2'].values
                y2_val = y_test['Roi.Y2'].values
                
                if self.__is_test_set_img:
                    try:
                        img_bbx = img.copy()
                        x1, y1, x2, y2 = util.get_roi(self.en_input_file.get(), x1_val, y1_val, x2_val, y2_val)
                        proposal = img[y1:y2, x1:x2]
                        result = util.recognize_sign([proposal], self.__model)[0]
                        sign_name = util.load_name(result, self.__label_values)
                        if len(sign_name) > 0:
                            # wm = plt.get_current_fig_manager()
                            # wm.window.showMaximized()
                            plt.imshow(cv2.cvtColor(img_bbx, cv2.COLOR_BGR2RGB))
                            plt.axis("off")
                            plt.title("Result of prediction: " + sign_name)
                            plt.show()
                        else:
                            messagebox.showinfo("Infomation", "Sorry. Can not recognize any traffic sign in this image.")
                    except Exception as ex:
                        print(ex)
                        messagebox.showinfo("Infomation", "Sorry. Can not recognize any traffic sign in this image.")
                else:
                    # convert Image to Binary Image
                    img_bbx = img.copy()
                    rows, cols, _ = img.shape
                    img_bin = util.preprocess_img(img, False)
        
                    # localize Traffic Sign (find Contours and draw to Image)
                    min_area = img_bin.shape[0]*img.shape[1]/(25*25)
                    rects = util.detect_contour(img_bin, min_area=min_area)
                    img_rects = util.draw_rects_on_img(img, rects)
        
                    sign_names = []
                    sep = ', '
                    
                    # recognize Traffic sign
                    for rect in rects:
                        xc = int(rect[0] + rect[2]/2)
                        yc = int(rect[1] + rect[3]/2)
                        size = max(rect[2], rect[3])
                        x1 = max(0, int(xc-size/2))
                        y1 = max(0, int(yc-size/2))
                        x2 = min(cols, int(xc + size/2))
                        y2 = min(rows, int(yc + size/2))
                        proposal = img[y1:y2, x1:x2]
                        result = util.recognize_sign([proposal], self.__model)[0]
                        cv2.rectangle(img_bbx, (rect[0],rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255), 2)
                        cv2.putText(img_bbx, str(result), (rect[0], rect[1]), 1, 1.5, (0, 0, 255), 2)
                        cv2.putText(img_bbx, util.load_name(result, self.__label_values), (rect[0], rect[1] + rect[3] + 20), 1, 1.5, (0, 0, 255), 2)
                        sign_names.append(util.load_name(result, self.__label_values))
                        
                    if len(sign_names) > 0:
                        sep_res = sep.join(sign_names).replace('/', '-')
                        time = datetime.now().strftime("%Y-%m-%d %Hh-%Mm-%Ss")
                        dir_path = cons.RESULT_PATH + time
                        util.create_directory(dir_path)
                        res_path = dir_path + '\\{}_' + sep_res.replace(',','-') + '.jpg'
                        cv2.imwrite(res_path.format("BIN_IMAGE"), img_bin)
                        print("[INFO]: Saved binary image to " + res_path.format("BIN_IMAGE"))
                        cv2.imwrite(res_path.format("RECTS_IMAGE"), img_rects)
                        print("[INFO]: Saved rects image to " + res_path.format("RECTS_IMAGE"))
                        cv2.imwrite(res_path.format("PREDICTED_IMAGE"), img_bbx)
                        print("[INFO]: Saved predicted image to " + res_path.format("PREDICTED_IMAGE"))
                        # wm = plt.get_current_fig_manager()
                        # wm.window.showMaximized()
                        plt.imshow(cv2.cvtColor(img_bbx, cv2.COLOR_BGR2RGB))
                        plt.axis("off")
                        plt.title("Result of prediction: " + sep_res)
                        plt.show()
                        messagebox.showinfo("Infomation", "Finish. Your result is saved in {}\\".format(dir_path))
                    else:
                        messagebox.showinfo("Infomation", "Sorry. Can not recognize any traffic sign in this image.")
                
                print("[INFO]: FINISH PREDICTING IMAGE.")
        except Exception as ex:
            print(ex)
            messagebox.showerror("Error", "Error occurred while predicting image!")
            
    def exit_app(self):
        try:
            msg = messagebox.askquestion('Exit Application','Are you sure to exit?',icon = 'info')
            if msg == 'yes':
                self.destroy()
        except Exception as ex:
            print(ex)
            messagebox.showerror("Error", "Error occurred while exiting the application!")
    
    def clear_all(self):
        try:
            if len(self.en_input_file.get()) > 0 or len(self.en_model_file.get()) > 0:
                msg = messagebox.askquestion('Clear Action','Are you sure to clear?',icon = 'info')
                if msg == 'yes':
                    self.lb_img_info.config(text='')
                    self.__is_test_set_img = False
                    self.__input_path = ''
                    self.__model_path = ''
                    self.en_input_file.config(state='normal')
                    self.en_model_file.config(state='normal')
                    self.en_input_file.delete('0', END)
                    self.en_model_file.delete('0', END)
                    self.en_input_file.config(state='readonly')
                    self.en_model_file.config(state='readonly')
        except Exception as ex:
            print(ex)
            messagebox.showerror("Error", "Error occurred while clearing!")