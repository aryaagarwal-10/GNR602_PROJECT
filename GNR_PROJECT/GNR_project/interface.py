import cv2 as cv
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from project import *
import os
from PIL import ImageTk, Image





op_file_path = 'del.png'
def clear_screen():
    for widget in root.winfo_children():
        widget.destroy()


def file_upload():
    input_1 = int(input_entry_1.get())
    input_2 = int(input_entry_2.get())
    input_4 = int(input_entry_4.get())
    input_5 = int(input_entry_5.get())
    input_6 = int(input_entry_6.get())
    input_7 = int(input_entry_7.get())
    input_8 = int(input_entry_8.get())
    input_3 = int(input_entry_3.get())
    input_9 = int(input_entry_9.get())
    input_0 = int(input_entry_0.get())
    
    
    global ip_file_path, op_file_path, img_ip, img_op
    ip_file_path = filedialog.askopenfilename()
    ip_file_path = os.path.abspath(ip_file_path)
    print("input_file:", ip_file_path)

    
    



    k=mainf(input_1,input_2,input_3,input_4,input_5,input_6,input_7,input_8,ip_file_path,input_9,input_0)

 

    clear_screen()
    #bar= ttk.Progressbar(root,orient=HORIZONTAL,length=300)
    #bar['value'] = bval
    #clear_screen()
    root.config(bg='#000000',height=750, width=1200)
    var = StringVar()
    label = Label( root, textvariable=var )

    var.set(str(k))
    label.pack()
    #kl = Label(root, text=str(k),fg='white', bg='white', font=("Arial", 24))
    #kl.place(x=825, y=120)
    op_file_path1='/Users/sdl/Desktop/GNR_project/output.jpeg'
    op_file_path2='/Users/sdl/Desktop/GNR_project/PCA1.jpeg'
    op_file_path3='/Users/sdl/Desktop/GNR_project/PCA1_without_normalizing.jpeg'


    nameList = ["/Users/sdl/Desktop/GNR_project/output.jpeg", "/Users/sdl/Desktop/GNR_project/PCA1.jpeg","/Users/sdl/Desktop/GNR_project/PCA1_without_normalizing.jpeg","/Users/sdl/Desktop/GNR_project/output2.jpeg"]
    i=0
    label=["output with optimal k","PCA1 after normalizing","PCA1 without normalizing","output with given k"]
    labelx=[20,420,820,20,420,820]
    labely=[50,50,50,420,450,450]
    yc=[80,80,80,450,450,480]
    xc=[30,430,830,30,430,830]
    for execute in nameList:
        myimg=Image.open(execute)
        myimg= myimg.resize((300,300), Image.ANTIALIAS)
        myimg = ImageTk.PhotoImage(myimg)
        lbl = Label(root, image=myimg)
        lbl.place(x=xc[i], y=yc[i])
        lbl.image = myimg 
        sl = Label(root, text=str(label[i]),fg='white', bg='black', font=("Arial", 24))
        sl.place(x=labelx[i], y=labely[i])
        i=i+1




 



root = Tk()
root.title("GNR602 Project Group no.57")
root.config(bg='#000020',height=750, width=1200)

# Input Parameters
input_label_1 = Label(root, text="maxi",fg='white', bg='black', font=("Arial", 24))
input_label_1.place(x=205, y=150)
input_entry_1 = Entry(root,font=("Arial", 24))
input_entry_1.place(x=350, y=150, width=100, height=50)

input_label_2 = Label(root, text="window size:",fg='white', bg='black', font=("Arial", 24))
input_label_2.place(x=525, y=150)
input_entry_2 = Entry(root,font=("Arial", 24))
input_entry_2.place(x=670, y=150, width=100, height=50)


input_label_4 = Label(root, text="glcm size:",fg='white', bg='black', font=("Arial", 24))
input_label_4.place(x=205, y=250)
input_entry_4 = Entry(root,font=("Arial", 24))
input_entry_4.place(x=350, y=250, width=100, height=50)

input_label_5 = Label(root, text="initial1:",fg='white', bg='black', font=("Arial", 24))
input_label_5.place(x=525, y=250)
input_entry_5 = Entry(root,font=("Arial", 24))
input_entry_5.place(x=670, y=250, width=100, height=50)

input_label_6 = Label(root, text="initial2:",fg='white', bg='black', font=("Arial", 24))
input_label_6.place(x=205, y=350)
input_entry_6 = Entry(root,font=("Arial", 24))
input_entry_6.place(x=350, y=350, width=100, height=50)

input_label_7 = Label(root, text="size1:",fg='white', bg='black', font=("Arial", 24))
input_label_7.place(x=525, y=350)
input_entry_7 = Entry(root,font=("Arial", 24))
input_entry_7.place(x=670, y=350, width=100, height=50)

input_label_8 = Label(root, text="size2:",fg='white', bg='black', font=("Arial", 24))
input_label_8.place(x=205, y=450)
input_entry_8 = Entry(root,font=("Arial", 24))
input_entry_8.place(x=350, y=450, width=100, height=50)

input_label_3 = Label(root, text="clusters:",fg='white', bg='black', font=("Arial", 24))
input_label_3.place(x=525, y=450)
input_entry_3 = Entry(root,font=("Arial", 24))
input_entry_3.place(x=670, y=450, width=100, height=50)

input_label_9 = Label(root, text="d1:",fg='white', bg='black', font=("Arial", 24))
input_label_9.place(x=205, y=550)
input_entry_9 = Entry(root,font=("Arial", 24))
input_entry_9.place(x=350, y=550, width=100, height=50)

input_label_0 = Label(root, text="d2:",fg='white', bg='black', font=("Arial", 24))
input_label_0.place(x=525, y=550)
input_entry_0 = Entry(root,font=("Arial", 24))
input_entry_0.place(x=670, y=550, width=100, height=50)


# Upload Button
button = Button(root, text="Upload", command=file_upload,font=("Arial", 20),relief=RIDGE)
button.place(x=450, y=650, width=100, height=50)






root.mainloop()