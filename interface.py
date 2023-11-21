from email.mime.text import MIMEText
from io import BytesIO, StringIO
import smtplib
from sklearn.decomposition import PCA
import streamlit as st
from streamlit_option_menu import option_menu
from email.header import Header
from PIL import Image
import os
import cv2
from A.data_preprocessing import load_data_log4A
from B.data_preprocessing import load_data_log4B

from utils import load_data, load_model
import numpy as np

st.set_page_config(page_title="DDI")  # system name
st.markdown(
    """
    <style>
    body {
        primary-color: #FF4B4B;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:  # sidebar of the system
    choose = option_menu("DDI", ["Welcome", "Diagnosis", "Feedback", "Help"],
                            menu_icon="hospital",
                            icons=['compass', 'clipboard2-pulse', 'envelope', 'question-circle'],
                            default_index=0)
    
if choose == "Welcome":  # instruction page
    st.title("🎊 Welcome to DDI!")
    st.write(
        f"Hi, welcome to the digital system of Disease Diagnosis with Image (DDI). This is the instruction page for DDI, a digital system supported by AI solutions. New patients of this app can refer to the "
        "following sections as guidelines. 👇")

    st.header("Diagnosis")
    with st.expander("See details", expanded=True):
        st.write("This is where you can make digital diagnosis for possible diseases with images.")
        st.subheader("👉 Want to diagnose for pneumonia? -- See our *Pneumonia* section.")
        st.write("• The **Pneumonia** takes your uploaded *:blue[chest X-ray slides]* for diagnosis. Please ensure that your file format is valid and clear "
                 "to guarantee accurate result.")

        st.subheader("👉 Feel hard to classify for hematoxylin & eosin stained histological tissue images? -- *CRC* helps.")
        st.write("• Upload your tissue image.")
        st.write("• Make digital classification for *:blue[9 types]* of tissue (ADI, BACK, DEB, etc).")
        st.write("• Get your tissue classification result supported by AI solutions!")


    st.header("Feedback")
    with st.expander("See details", expanded=True):
        st.write("This is the place for medical advice and additional diagnostic info survey.")
        st.subheader("👉 Want to check medical advice from specialists? -- *Advice* helps.")
        st.write(
            "• **Specialists for each disease** can leave medical advice for patients accompanied with AI diagnosis.")
        st.write("• Your medical advice will be sent as *:blue[automatic system email]*. 📬")

        st.subheader("🙌 Additional diagnostic survey for comprehensive knowledge of your condition!")
        st.write("• We appreciate your time for providing additional infomation on your recent health condition, "
                 "which can give more insights for specialists to diagnose.")

    st.header("Help")
    with st.expander("See details", expanded=True):
        st.write("Contact **DDI Developer Team** if you have any problems or suggestions. Glad to see your "
                    "contribution.")



elif choose == "Diagnosis":
    disease = option_menu(None, ["Pneumonia", "CRC"],
                               icons=['lungs', 'eyedropper'], default_index=0, orientation="horizontal")
    # 可以放一些说明images，写论文的时候一起
    if disease == "Pneumonia":
        uploaded_image = st.file_uploader("Choose an image for diagnosis.", accept_multiple_files=False)
        if uploaded_image != None:
            st.download_button(f'Download {uploaded_image.name}', uploaded_image, mime="image/png")
        
            bytes_data = uploaded_image.getvalue() # To read file as bytes
            # print(bytes_data)
            bytes_stream = BytesIO(bytes_data)  # convert bytes into stream
            user_img = Image.open(bytes_stream)

            imgByteArr = BytesIO()    # 初始化一个空字节流
            user_img.save(imgByteArr,format('PNG'))     #把我们得图片以‘PNG’保存到空字节流
            imgByteArr = imgByteArr.getvalue()    #无视指针，获取全部内容，类型由io流变成bytes。
            if not os.path.exists("Outputs/images/interface/"):
                os.makedirs("Outputs/images/interface/") 
            with open("Outputs/images/interface/user_img.png",'wb') as f:
                f.write(imgByteArr)
            
            user_img = cv2.imread("Outputs/images/interface/user_img.png")  # open with cv2 to guarantee consistency for subsequent processing

            col1, col2 = st.columns(2)
            with col1:
                st.image(user_img,width=200)

            with col2:
                with st.status("Diagnose for pneumonia condition..."):
                    single_img = cv2.cvtColor(user_img,cv2.COLOR_BGR2GRAY)
                    h,w = np.array(single_img).shape
                    single_img = np.array(single_img).reshape(1,h*w)
                    # notice that you can run this streamlit application only when you first got the dataset stored
                    print(f"Method: SVM Task: A.")
                    pre_path = 'Outputs/pneumoniamnist/preprocessed_data'
                    load_data_log4A() 
                    print("Start loading data......")
                    Xtrain, ytrain, Xtest, ytest, Xval, yval = load_data("A",pre_path,"SVM")
                    print("Load data successfully.")
                    print("Start loading model......")
                    model = load_model("A", "SVM")
                    print("Load model successfully.")
                    
                    model.train(Xtrain, ytrain, Xval, yval)
                    pred_train, pred_val, pred_test = model.test(Xtrain, ytrain, Xval, yval, single_img)
                    if pred_test[0] == "1":
                        st.error("Your diagnosis result is: penumonia.")
                    else:
                        st.success('Your diagnosis result is: non-penumonia.')

                st.button('Rerun')
            
    elif disease == "CRC":
        uploaded_image = st.file_uploader("Choose an image for diagnosis.", accept_multiple_files=False)
        if uploaded_image != None:
            st.download_button(f'Download {uploaded_image.name}', uploaded_image, mime="image/png")
        
            bytes_data = uploaded_image.getvalue() # To read file as bytes
            # print(bytes_data)
            bytes_stream = BytesIO(bytes_data)  # convert bytes into stream
            user_img = Image.open(bytes_stream)

            imgByteArr = BytesIO()    # 初始化一个空字节流
            user_img.save(imgByteArr,format('PNG'))     #把我们得图片以‘PNG’保存到空字节流
            imgByteArr = imgByteArr.getvalue()    #无视指针，获取全部内容，类型由io流变成bytes。
            if not os.path.exists("Outputs/images/interface/"):
                os.makedirs("Outputs/images/interface/") 
            with open("Outputs/images/interface/user_img.png",'wb') as f:
                f.write(imgByteArr)
            
            user_img = cv2.imread("Outputs/images/interface/user_img.png")  # open with cv2 to guarantee consistency for subsequent processing

            col1, col2 = st.columns(2)
            with col1:
                st.image(user_img,width=200)

            with col2:
                with st.status("Classify for tissue types..."):
                    # h,w,c = np.array(user_img).shape
                    # single_img = np.array(user_img).reshape(1,h*w*c)
                    # uploaded_image.name.split("_")
                
                    # notice that you can run this streamlit application only when you first got the dataset stored
                    print(f"Method: NB Task B.")
                    pre_path = 'Outputs/pathmnist/preprocessed_data'
                    load_data_log4B() 
                    print("Start loading data......")
                    Xtrain, ytrain, Xtest, ytest, Xval, yval = load_data("B",pre_path,"NB")
                    single_img = np.array(Xtest[int(uploaded_image.name.split("_")[0][4:]),:]).reshape(1,64)
                    print("Load data successfully.")
                    print("Start loading model......")
                    model = load_model("B", "NB")
                    print("Load model successfully.")
                    
                    model.train(Xtrain, ytrain, Xval, yval)
                    pred_train, pred_val, pred_test = model.test(Xtrain, ytrain, Xval, yval, single_img)
                    type = {
                        "0":"ADI","1":"BACK","2":"DEB","3":"LYM",\
                        "4":"MUC","5":"MUS","6":"NORM","7":"STR","8":"TUM"
                        }
                    st.success(f'Your diagnosis result is: {type[pred_test[0]]}.')

                st.button('Rerun')

elif choose == "Feedback":
    st.title("Feedback")
    feedback = option_menu(None, ["Advice", "Survey"],
                            icons=['person-badge', 'clipboard-check'], default_index=0,
                            orientation="horizontal")

    if feedback == "Advice":
        # teacher side: teacher leave message for student
        # student leave message for others
        from_addr = "2387324762@qq.com"
        password = 'pdewxqltfshtebia'
        to_addr = st.text_input("Send message to", placeholder="xxxxxxxxxx@qq.com")
        smtp_server = 'smtp.qq.com'
        with st.form("message"):
            message = st.text_area(label="", placeholder="Leave your message here...",
                                    label_visibility="collapsed")
            send = st.form_submit_button("Send")
            if send:
                name = "Dr.Bethune"
                msg = MIMEText(f'Dear Madam/Sir,\n' + '  ' + message + '\n'
                                                                        '\n'
                                                                        'Best regards,\n'
                                                                        f'{name} from DDI', 'plain', 'utf-8')
                msg['From'] = Header(from_addr)
                msg['To'] = Header(from_addr)
                subject = f'DDI: Message from {name}'
                msg['Subject'] = Header(subject, 'utf-8')

                try:
                    smtpobj = smtplib.SMTP_SSL(smtp_server)
                    smtpobj.connect(smtp_server, 465)
                    smtpobj.login(from_addr, password)
                    smtpobj.sendmail(from_addr, to_addr, msg.as_string())
                    print("Send successfully")
                except smtplib.SMTPException:
                    print("Fail to send")
                finally:
                    smtpobj.quit()
                st.success("Your message is sent successfully.")

    elif feedback == "Survey":
        with st.form("survey"):
            col1, col2 = st.columns(2)
            with col1:
                st.radio(
                    "Q1: Do you feel loss of appetite or restlessness?",
                    ["Yes", "No"],
                )
                st.radio(
                    "Q3: Do you find it difficult breathing recently?",
                    ["Very difficult", "More difficult than usual", "Not quite"],
                )
            with col2:
                st.radio(
                    "Q2: Do you get fever recently?",
                    ["Yes", "No"],
                )
                st.radio(
                    "Q4: Do you feel abdomen painfully swollen recently?",
                    ["Very frequently", "Not frequently", "Not quite"],
                )

            st.selectbox('Q5: What is your frequency of coughing?', (
                'Don\'t cough', 'One to two times per day', 'More than 5 times', 'Frequently for each hour',
            ))

            st.options = st.multiselect(
                'Q6: What other symptoms do you have?',
                ['Pale complexion', 'Emesis',"Diarrhoea","hematochezia","Anemia"])

            st.slider('Q7: How much weight have you lost (in kg)?', 0, 20, 0)

            submit = st.form_submit_button("Submit")
            if submit:
                st.success("Thank you for your consultation.")

elif choose == "Help":
    st.title("Help")
    with st.form("help"):
        col1, col2 = st.columns(2)
        with col1:
            st.text("\n")
            st.markdown("**Official email address:**")
            for i in range(2): st.text("\n")
            st.markdown("**Official tel:**")
        with col2:
            st.code("DDI_official2023@163.com", language='markdown')
            st.code("+44-7551167050", language='markdown')

        from_addr = "2387324762@qq.com"
        password = 'pdewxqltfshtebia'
        to_addr = '2387324762@qq.com'
        smtp_server = 'smtp.qq.com'
        message = st.text_area(label="", placeholder="Leave your problems or suggestions here...",
                                label_visibility="collapsed")
        contact = st.form_submit_button("Contact")
        if contact:
            msg = MIMEText('Dear DDI Developer Team,\n' + '  ' + message + '\n'
                                                                                '\n'
                                                                                'Best regards,\n'
                                                                                'Message from DDI', 'plain',
                            'utf-8')
            msg['From'] = Header(from_addr)
            msg['To'] = Header(from_addr)
            subject = f'DDI: Help & Contact message'
            msg['Subject'] = Header(subject, 'utf-8')

            try:
                smtpobj = smtplib.SMTP_SSL(smtp_server)
                smtpobj.connect(smtp_server, 465)
                smtpobj.login(from_addr, password)
                smtpobj.sendmail(from_addr, to_addr, msg.as_string())
                print("Send successfully")
            except smtplib.SMTPException:
                print("Fail to send")
            finally:
                smtpobj.quit()
            st.balloons()
            st.success("Your message is sent successfully.")
