import re
import json
from tqdm import tqdm
from unidecode import unidecode
from requests_html import HTMLSession
from bs4 import BeautifulSoup

from utils import currentTime


numreg = re.compile(r'[0-9]+')

def reName(string):
    '''
    Chuẩn hoá tên các thuộc tính: Loại bỏ dấu, khoảng trắng, in hoa chữ cái đầu.
    VD "thuộc tính" >> "ThuocTinh"
    '''
    return ''.join([w.capitalize() for w in unidecode(string).split()])


def preClean(paragraph):
    '''
    Xử lý văn bản. Loại bỏ kí tự 'non-breaking space', thay thế ngoặc kép bằng ngoặc
    đơn, xoá bỏ khoảng trắng dư thừa, kiểm tra forwad-slash cuối câu
    '''
    paragraph = paragraph.replace(u'\xa0', u' ')
    paragraph = paragraph.replace('\"', '\'')

    paragraph = re.sub(' +', ' ', paragraph)

    if paragraph.endswith('\\'):
        paragraph += '.' 

    return paragraph


def getdata(url):
    '''
    Crawl các thông tin về bất động sản hiện tại từ URL đến bài đăng bán cho trước.
    Dữ liệu trả về là status code của request và dữ liệu thu được dưới dạng (<statuscode>, dict(<data>))
    '''
    data = {}
    session = HTMLSession()
    res = session.get(url)
    status = res.status_code
    if status == 200:
        try:
            soup = BeautifulSoup(res.text, 'html.parser')
            data['NgayDangBan'] = soup.select_one('div.text-gray.text-100').get_text().strip()

            user = soup.select_one('div.show-user-info > div > div')
            id_re = re.compile(r'https:\/\/nhadatvui.vn\/user\/\s*(.*)')
            data['Id_NguoiDangban'] = re.findall(id_re, user.a['href'])[0]
            data['NguoiDangban'] = user.a.string.strip()

            crumb = soup.select_one('ul.crumb').select('li')
            data['LoaiBDS'] = crumb[1].span.string.strip()
            data['Tinh']    = crumb[2].span.string.strip()
            data['Huyen']   = crumb[3].span.string.strip()
            data['Xa']      = crumb[4].span.string.strip()

            propertyTitlePrice = soup.select_one('div.mt-3.product-title-price > div').select('span')
            data['DiaChi']  = propertyTitlePrice[0].string.strip()
            data['TongGia'] = propertyTitlePrice[1].string.strip()

            try:
                data['Gia/m2'] = propertyTitlePrice[2].string.strip()
            except:
                data['Gia/m2'] = '--'

            data['MaTin'] = soup.select_one('div.product-show-left > div.pt-6.pb-6.border-t.border-b > div > div:nth-child(4) > span:nth-child(2)').string.strip()

            propertyInfo = soup.select_one('ul.list-full-info-product').select('li')
            for info in propertyInfo:
                infoName = reName(info.select('span')[1].string.strip())
                data[infoName] = preClean(info.select('span')[2].string.strip()).replace('\n','')
            
            propertyDetail = soup.select_one('#content-tab-info div div ul').select('li')
            for info in propertyDetail:
                data[reName(info.span.span.string)] = " ".join(info.span.find_next_sibling('span').string.strip().split())
            try:
                del data['DienTichSuDung']
            except:
                pass

            propertyUtil = soup.select_one('div.tabs-product').select('ul.product-other-utilities')
            propertyUtilHead = soup.select_one('div.tabs-product').select('h2')
            for _, util in enumerate(propertyUtilHead[2:]):
                utilName = reName(util.string.strip())
                data[utilName] = []
                for item in propertyUtil[_].select('li'):
                    data[utilName].append(item.span.string)
        except Exception as e:
            print()
            print('>>> Error: %s.' %(e))
            try:
                print('Post', data['MaTin'])
            except:
                try:
                    MaTin = soup.select_one('.product-title-price div')\
                        .findChildren('div',recursive=False)[1]\
                        .select('.product-status div')[0]\
                        .select('span')[1].string.strip()
                    print('Post', MaTin)
                except:
                    print("Can't get MaTin")
            pass
    else:
        print('>>> Error: Request URL fail. Status code', status)
    return status, data


def appendData(idx, current_row, data, url_data):
    '''
    Thực thi hàm getdata() crawl dữ liệu từ URL hiện tại, cập nhật trạng thái URL, thời gian crawl
    '''
    status, property_data = getdata(current_row['url'])

    if status == 200:
        property_data['id'] = current_row['id']
        data.append(property_data)
        url_data.loc[idx, 'crawlingDate'] = currentTime()

    else:
        print('-- Post %d.' %(current_row['id']))
        url_data.loc[idx, 'status'] = 'unavailable'
        
    url_data.loc[idx, 'crawled'] = 1

    return data, url_data


def crawlData(url_data, recrawl = False, from_idx = 0, to_idx = None):
    '''
    Duyệt hàm appendData() qua danh sách URL cần crawl.
    * recrawl = True: crawl lại từ toàn bộ liên kết trong danh sách URL
    * recrawl = False: chỉ crawl từ các URL mới (url_data['crawled'] == 0)
    '''
    if to_idx is None:
        to_idx = len(url_data)
    url_data_crawl = url_data[from_idx:to_idx].copy()
    data = []

    if recrawl == True:
        for idx, row in tqdm(url_data_crawl.iterrows(), total=url_data_crawl.shape[0]):
            data, url_data_crawl = appendData(idx, row, data, url_data_crawl)
            if idx % 500 == 0:
                path = 'data/raw/temp_%s.json' %(idx)
                with open(path, "w") as outfile:
                    outfile.write(json.dumps(data, indent=2, ensure_ascii=False))
                print("Save temp to ./%s" %(path))

    elif recrawl == False:
        for idx, row in tqdm(url_data_crawl.iterrows(), total=url_data_crawl.shape[0]):
            if row['crawled'] == 1:
                continue
            else:
                data, url_data_crawl = appendData(idx, row, data, url_data_crawl)
                if (idx+1) % 1000 == 0:
                    path = 'data/raw/temp/temp_%s.json' %(idx+1)
                    with open(path, "w") as outfile:
                        outfile.write(json.dumps(data, indent=2, ensure_ascii=False))
                    print("Save temp to ./%s" %(path))

    url_data.update(url_data_crawl)
    url_data = url_data.astype({'id':'int64','crawled':'int64'})

    return data, url_data


def saveDataCSV(df, overwrite = False) -> str:
    if overwrite == False:
        filename = 'raw_' + currentTime().strftime("%Y%m%d_%H%M") +'.csv'
    else:
        filename = 'raw.csv'

    path = 'data/raw/' + filename
    df.to_csv(path, index=False, header=True)
    print("Save dataframe to ./%s" %(path))
    
    return path


def saveDataJSON(postData, overwrite = False) -> str:
    if overwrite == False:
        filename = 'raw_' + currentTime().strftime("%Y%m%d_%H%M") +'.json'
    else:
        filename = 'raw.json'

    path = 'data/raw/' + filename
    with open(path, "w") as outfile:
        outfile.write(json.dumps(postData, indent=2, ensure_ascii=False))
    print("Save data to ./%s" %(path))
    
    return path