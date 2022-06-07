import re
import win32com.client
from bs4 import BeautifulSoup
from uti import timeit


def outlook_initialize(folder_name='GS'):
    outlook = win32com.client.Dispatch('outlook.application').GetNamespace("MAPI")
    accounts = win32com.client.Dispatch("Outlook.Application").Session.Accounts

    for account in accounts:

        folders = outlook.Folders(account.DeliveryStore.DisplayName)
        specific_folder = folders.Folders

        for folder in specific_folder:
            if folder.name == folder_name:
                messages = folder.Items
                return messages

    raise Exception('GS folder and message not found, please check.')


@timeit
def outlook_get_marquee_link(messages):
    for message in messages:
        soup = BeautifulSoup(message.HTMLBody, 'html.parser')

        pattern = re.compile('<a href="(https://.*marquee.gs.com.*)"><.*>')

        links = []
        for class_a in soup.findAll('a'):
            if re.match(pattern, str(class_a)):
                links.append(re.findall(pattern, str(class_a))[0])

        if len(links) > 0:
            return links
        return None

