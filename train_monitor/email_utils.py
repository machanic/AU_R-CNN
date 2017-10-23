import smtplib
from email.mime.text import MIMEText
from contextlib import closing


class EmailUtils(object):
    @classmethod
    def sendMail(self, to_list=[], sub='', content=''):
        mail_host = "mail.163.com";
        mail_user = "sharpstill@163.com"
        mail_pass = "sinabaofeng1998"
        mail_postfix = "163.com"
        me = mail_user + "<" + mail_user + "@" + mail_postfix + ">";
        msg = MIMEText(content, "html", _charset='utf-8')
        msg['Subject'] = sub;
        msg['From'] = me
        msg['To'] = ";".join(to_list)
        try:
            with closing(smtplib.SMTP_SSL("smtp.163.com", 465)) as mail:
                mail.set_debuglevel(1)
                mail.login(mail_user, mail_pass)
                mail.sendmail(mail_user, to_list, msg.as_string())
                mail.close()
            return True
        except Exception:
            return False


if __name__ == '__main__':
    for i in range(3):
        EmailUtils.sendMail(["sharpstill@163.com"], "python test", "python hello world")
