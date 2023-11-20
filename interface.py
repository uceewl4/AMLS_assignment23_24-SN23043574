from email.mime.text import MIMEText
import smtplib
import streamlit as st
from streamlit_option_menu import option_menu
from email.header import Header

st.set_page_config(page_title="DDI")  # system name

with st.sidebar:  # sidebar of the system
    choose = option_menu("DDI", ["Welcome", "Diagnosis", "Feedback", "Help"],
                            menu_icon="mortarboard-fill",
                            icons=['compass', '', 'envelope', 'question-circle'],
                            default_index=0)
    
if choose == "Welcome":  # instruction page
    st.title("🎊 Welcome to DDI!")

elif choose == "Diagnosis":
    disease = option_menu(None, ["Pneumonia", "CRC"],
                               icons=['person-square', 'calendar3'], default_index=0, orientation="horizontal")
    # 可以放一些说明images，写论文的时候一起
    if disease == "Peneumonia":
        uploaded_image = st.file_uploader("Choose an image", accept_multiple_files=False)
        st.download_button(f'Download {uploaded_image.name}', uploaded_image, mime="image/png")

        # image = Image.open('sunrise.png')

        # st.image(image, caption='Sunrise by the mountains')
    else:
        pass



elif choose == "Feedback":
    st.title("Feedback")
    feedback = option_menu(None, ["Message", "Survey"],
                            icons=['chat-left-text', 'clipboard-check'], default_index=0,
                            orientation="horizontal")

    if feedback == "Message":
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
                name = "Dr."
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
                    "Q1: Do you like making study plans and follow them?",
                    ["Yes", "No"],
                )
                st.radio(
                    "Q3: What might be the influence when you are compared with others?",
                    ["More motivated", "More stressful", "No influence"],
                )
            with col2:
                st.radio(
                    "Q2: Do you think setting goals for your study is inspiring?",
                    ["Yes", "No"],
                )
                st.radio(
                    "Q4: Which one do you prefer, study alone or in group?",
                    ["Study alone", "Study in group", "Both"],
                )

            st.selectbox('Q5: From your perspective, the most important assignment or performance is', (
                'Labs', 'Class test', 'Question 1', 'Question 2',
                'Question 3', 'Question 4', 'Exam', 'QMUL or BUPT result', 'Engagement', 'Attendance %',
                'Video views', 'Grade and whether at-risk'
            ))

            st.options = st.multiselect(
                'Q6: Which course do you think is difficult?',
                ['EBU6501 Middleware', 'EBU5476 Microprocessor'])

            st.slider('Q7: How many hours do you usually spend on studying?', 0, 24, 0)
            st.slider('Q8: The extent of satisfaction for your current performance record is', 0, 5, 0)

            submit = st.form_submit_button("Submit")
            if submit:
                st.success("Thank you for your participation.")

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
