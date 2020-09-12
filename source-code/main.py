from gui import gui_obj

if __name__ == '__main__':
    try:
        root = gui_obj()
        root.mainloop()
    except Exception as ex:
        print(ex)
        print("Error occurred while executing!!!")