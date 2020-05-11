
import pandas as pd
import requests             
from bs4 import BeautifulSoup 
import csv                  
import webbrowser
import io

def display(content, filename='output.html'):
    with open(filename, 'wb') as f:
        f.write(content)
        webbrowser.open(filename)

def get_soup(session, url):
    r = session.get(url)
    return BeautifulSoup(r.text, 'html.parser')
    
def post_soup(session, url, params):
    '''Read HTML from server and convert to Soup'''
    r = session.post(url, data=params)
    return BeautifulSoup(r.text, 'html.parser')
    
def scrape(url, lang='ALL'):

    # create session to keep all cookies (etc.) between requests
    session = requests.Session()

    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0',
    })

    items = parse(session, url + '?filterLang=' + lang)

    return items

def parse(session, url):
    '''Get number of reviews and start getting subpages with reviews'''

    print('[parse] url:', url)

    soup = get_soup(session, url)
    num_reviews = 200
    print('[parse] num_reviews ALL:', num_reviews)

    url_template = url.replace('.html', '-or{}.html')
    print('[parse] url_template:', url_template)

    items = []

    offset = 0
    while(True):
        subpage_url = url_template.format(offset)

        subpage_items = parse_reviews(session, subpage_url)
        if not subpage_items:
            break

        items += subpage_items

        if len(subpage_items) < 5:
            break

        offset += 5

    return items

def get_reviews_ids(soup):

    items = soup.find_all('div', attrs={'data-reviewid': True})

    if items:
        reviews_ids = [x.attrs['data-reviewid'] for x in items]
        print('[get_reviews_ids] data-reviewid:', reviews_ids)
        return reviews_ids
    
def get_more(session, reviews_ids):

    url = 'https://www.tripadvisor.com/OverlayWidgetAjax?Mode=EXPANDED_HOTEL_REVIEWS_RESP&metaReferer=Hotel_Review'

    payload = {
        'reviews': ','.join(reviews_ids), # ie. "577882734,577547902,577300887",
        'contextChoice': 'DETAIL_HR', # ???
        'widgetChoice': 'EXPANDED_HOTEL_REVIEW_HSX', # ???
        'haveJses': 'earlyRequireDefine,amdearly,global_error,long_lived_global,apg-Hotel_Review,apg-Hotel_Review-in,bootstrap,desktop-rooms-guests-dust-en_US,responsive-calendar-templates-dust-en_US,taevents',
        'haveCsses': 'apg-Hotel_Review-in',
        'Action': 'install',
    }

    soup = post_soup(session, url, payload)

    return soup

def parse_reviews(session, url):
    '''Get all reviews from one page'''

    print('[parse_reviews] url:', url)

    soup =  get_soup(session, url)

    if not soup:
        print('[parse_reviews] no soup:', url)
        return

    #name_of_hotel = soup.find('h1', id='HEADING').getText()

    reviews_ids = get_reviews_ids(soup)
    if not reviews_ids:
        return

    soup = get_more(session, reviews_ids)

    if not soup:
        print('[parse_reviews] no soup:', url)
        return

    items = []

    for idx, review in enumerate(soup.find_all('div', class_='reviewSelector')):

        badgets = review.find_all('span', class_='badgetext')
        if len(badgets) > 0:
            contributions = badgets[0].text
        else:
            contributions = '0'

        if len(badgets) > 1:
            helpful_vote = badgets[1].text
        else:
            helpful_vote = '0'
        user_loc = review.select_one('div.userLoc strong')
        if user_loc:
            user_loc = user_loc.text
        else:
            user_loc = ''
            
        bubble_rating = review.select_one('span.ui_bubble_rating')['class']
        bubble_rating = bubble_rating[1].split('_')[-1]

        item = {'review_title': review.find('span', class_='noQuotes').text,
                'review_body': review.find('p', class_='partial_entry').text,
                'review_date': review.find('span', class_='ratingDate').text, # 'ratingDate' instead of 'relativeDate'
                'rating': (review.select_one('span.ui_bubble_rating')['class'])[1].split('_')[-1]
        }

        items.append(item)
        print('\n--- review ---\n')
        for key,val in item.items():
            print(' ', key, ':', val)

    print(items)

    return items

def write_in_csv(items, filename,headers):

    print('--- CSV ---')
    with open(filename, 'a') as csvfile:
        #tsv_writer = csv.DictWriter(tsvfile,headers, delimiter='\t')
        csv_file = csv.DictWriter(csvfile, headers)
        #tsv_writer.writerows(items)
        csv_file.writerows(items)

def main():
    DB_COLUMN   = 'review_title'
    DB_COLUMN1 = 'review_body'
    DB_COLUMN2 = 'review_date'
    DB_COLUMN3 = 'rating'

    headers = [ 
        DB_COLUMN, 
        DB_COLUMN1, 
        DB_COLUMN2,
        DB_COLUMN3,
    ]   


    start_urls = [ 'https://www.tripadvisor.com/Hotel_Review-g317114-d7809136-Reviews-Anjali_s_Dolphins_Resort-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d2069567-Reviews-Mr_B_s_Place-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d1604058-Reviews-Forest_Hideaway_Hotel_Cottages-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d2357645-Reviews-Bardia_Wildlife_Resort-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g12320054-d13280800-Reviews-Bardia_Wild_Planet_Jungle_Retreat-Thakurdwara_Bardia_National_Park_Bheri_Zone_Mid_W.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d2661058-Reviews-Jungle_Base_Camp-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d2660998-Reviews-Mango_Tree_Lodge-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d12847429-Reviews-Rhino_Lodge_Bardia-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d1961710-Reviews-Bardia_Kingfisher_Resort-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d2432667-Reviews-Bardia_Eco_Lodge-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d2317963-Reviews-Nature_Safari_Resort_Lodge-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g12320054-d7272090-Reviews-Wild_Trak_Lodge-Thakurdwara_Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d15579664-Reviews-Bardia_Community_Homestay-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g12320054-d18206781-Reviews-Bardia_Jungle_Resort-Thakurdwara_Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d394923-Reviews-Tiger_Tops_Karnali_Lodge-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d3748822-Reviews-Nature_Safari_Resort-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d5848994-Reviews-Jungle_Heaven_Hotel_Cottage-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g12320054-d8286741-Reviews-Samsara_Safari_Camp-Thakurdwara_Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d5408500-Reviews-Bardia_Tiger_Resort-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d1988800-Reviews-Samarth_Bardia_Adventure_Resort-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',
'https://www.tripadvisor.com/Hotel_Review-g12320054-d12661175-Reviews-SPOT_ON_638_Bardiya_Gaida_Camp-Thakurdwara_Bardia_National_Park_Bheri_Zone_Mid_West.html',
'https://www.tripadvisor.com/Hotel_Review-g317114-d775426-Reviews-Bardia_Jungle_Cottage-Bardia_National_Park_Bheri_Zone_Mid_Western_Region.html',]

    lang = 'en'

    for url in start_urls:
        # get all reviews for 'url' and 'lang'
        items = scrape(url, lang)
        filename = url.split('Reviews-')[1][:-5] + '.csv'
        print('filename:', filename)
        write_in_csv(items, filename, headers)




if __name__ == "__main__":
    main()





