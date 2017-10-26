#!/usr/bin/python

import smtplib

def send_email_notification(subject, text):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()

    from_addr = username
    to_addr = [username]

    server.login(username, password)

    message = "From: %s\nTo: %s\nSubject: %s\n\n%s" % (from_addr, ", ".join(to_addr),
            subject, text)

    server.sendmail(from_addr, to_addr, message)
