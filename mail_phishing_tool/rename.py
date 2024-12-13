import os
# change name of the email files into mail1, mail2, mail3, etc.
i = 51
for emails in os.listdir('Mails'):
    if emails.endswith('.eml'):
        src = 'Mails/' + emails
        dst = 'Mails/mail' + str(i) + '.eml'
        os.rename(src, dst)
        i += 1
