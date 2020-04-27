import os
import shutil
import requests
import urllib.request
from urllib.request import urlretrieve
from bs4 import BeautifulSoup

total_count = 0
i
y_m_url = 'http://www.getchu.com/all/month_title.html'

years = [str(y) for y in list(range(2019, 2020))]
months = [str(m).zfill(2) for m in list(range(1, 2))]

root_dir = './images'
# os.mkdir(root_dir)

payload = {
    'gage': 'all',
    'gc': 'gc' # important
}

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

types = ['pc_soft', 'dvd_game']

max_retries = 1
for y in years:
    ct = 1
    out_dir = os.path.join(root_dir, y)
    # os.mkdir(out_dir)
    for m in months:
        print("Scraping images in year {}, month {}".format(y, m))
        for t in types:
            success = False
            retries = 0
            while not success:
                try:
                    by_year_month_res = requests.get(y_m_url, params = {**payload, 'year': y, 'month': m, 'genre': t})
                    year_month_soup = BeautifulSoup(by_year_month_res.text, 'html.parser')
                    game_elems = year_month_soup.find_all('td', class_ = 'dd')
                    for game in game_elems:
                        game_ref = game.find('a').attrs['href']
                        game_url = root_url + game_ref

                        success = False
                        retries = 0
                        while not success:
                            try:
                                game_page_res = requests.get(game_url, params = {'gc': 'gc', 'Referer': 'http://www.getchu.com'})
                                game_page_soup = BeautifulSoup(game_page_res.text, 'html.parser')
                                img_tags = game_page_soup.find_all('img', attrs = { 'alt': lambda x : x and 'キャラ' in x})
                                print(img_tags)
                                character_tags = [root_url + tag.attrs['src'][1:] for tag in img_tags]
                                for character in character_tags:
                                    urlretrieve(character, os.path.join(out_dir, '{}_{}.jpg'.format(y, ct)))
                                    ct += 1
                                total_count += len(character_tags)
                                print("Total images: {}".format(total_count))
                                success = True
                            except:
                                print("Fetch url {} fail!".format(game_url))
                                retries += 1
                                if retries == max_retries:
                                    success = True
                    success = True
                except:
                    print("Fetch url {} fail!".format(y_m_url))
                    retries += 1
                    if retries == max_retries:
                        success = True