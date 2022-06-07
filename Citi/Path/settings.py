path = __file__
if 'jayyang' in path:
    path_len = 2
else:
    path_len = 3

if '\\' in path:
    BASE_PATH = '/'.join(path.split("\\")[:path_len])
else:
    BASE_PATH = '/'.join(path.split("/")[:path_len])


ONEDRIVE_PATH = BASE_PATH + "/OneDrive - Alpha Sherpa Capital"
PLATFORM_PATH = f"{BASE_PATH}/Platform"
DAILY_PATH = f"{ONEDRIVE_PATH}/Daily"
DATABASE_PATH = f"{ONEDRIVE_PATH}/Database/Citi"

print(__file__, path)
print('Base path', BASE_PATH)
print('Platform path', PLATFORM_PATH)
print('Onedrive directory:', ONEDRIVE_PATH)
print('Daily path', DAILY_PATH)
print('Database path', DATABASE_PATH)
