import pandas as pd
from main_utils.utils import currentTime
from requests_html import HTMLSession
from bs4 import BeautifulSoup
from tqdm import tqdm

def lastPage(url = "https://nhadatvui.vn/mua-ban/nha-dat") -> int:
    ''' Kiểm tra tổng số trang bài đăng hiện có thông qua STT trang cuối cùng trên pagination tab '''
    session = HTMLSession()
    res = session.get(url, headers={ 'User-Agent': 'Mozilla/5.0' })
    soup = BeautifulSoup(res.text, 'html.parser')
    last_page = soup.select('.pagination .page-item')[-2].get_text()
    return int(last_page)

def getPageURLs(start_url = "https://nhadatvui.vn/mua-ban/nha-dat", start_page = 1, end_page = None) -> list:
    ''' Crawl danh sách URL các 'page' duyệt từ trang chủ của nhadatvui '''
    if end_page is None:
        end_page = lastPage(start_url)
    return [f'{start_url}?page={page}' for page in range(start_page, end_page + 1)]

def getPostURLs(start_url = 'https://nhadatvui.vn/mua-ban/nha-dat', start_page = 1, end_page = None) -> list:
    ''' Crawl tất cả url chuyển tiếp tới tin bán bất động sản từ trang chủ của nhadatvui '''
    headers = { 'User-Agent': 'Mozilla/5.0' }
    data = []
    if end_page is None:
        end_page = lastPage(start_url)

    # Get list of all real estate sale post url
    for page_url in tqdm( getPageURLs(start_url, start_page, end_page) ):
        session = HTMLSession()
        res = session.get(page_url, headers=headers)

        # Parse the html content
        soup = BeautifulSoup(res.text, 'html.parser')
        # Get direct link to property
        property = soup.find_all("p", class_='name-product')

        for i in range(len(property)):
            follow_url = property[i].a['href']
            data.append(follow_url)
    return data

def URLToDF(list_URL: list) -> pd.DataFrame:
    ''''Create new dataframe from list of URL'''
    df_URL = pd.DataFrame(list_URL, columns=['url'])

    # Get id from URL
    df_URL['id'] = df_URL['url'].str[-10:]

    # Init URL default information 
    df_URL['crawled'] = 0
    df_URL['status'] = 'available'
    df_URL['crawlingDate'] = ''
    
    return df_URL

# Xuất file csv
def saveURLdf(df_URL, overwrite = False) -> str:
    if overwrite == False:
        filename = 'url_' + currentTime().strftime("%Y%m%d_%H%M") +'.csv'
    else:
        filename = 'url.csv'

    path = 'data/url/' + filename
    df_URL.to_csv(path, index=False, header=True)
    print("Save dataframe to ./%s" %(path))
    
    return path