from typing import Iterable, Any

import scrapy
from ..items import MatchItem
from scrapy_playwright.page import PageMethod


class FlashscoreSpider(scrapy.Spider):
    name = 'flashscore'
    start_urls = ['https://www.flashscore.ro/fotbal/romania/superliga-2024-2025/rezultate/',
                  'https://www.flashscore.ro/fotbal/romania/superliga-2023-2024/rezultate/',
                  'https://www.flashscore.ro/fotbal/romania/superliga-2022-2023/rezultate/',
                  'https://www.flashscore.ro/fotbal/romania/superliga-2021-2022/rezultate/',
                  'https://www.flashscore.ro/fotbal/romania/superliga-2020-2021/rezultate/',
                  ]
    async def load_more(self, page):
        load_more_button_locator = page.locator(
            'xpath=//a[@class="wclButtonLink wcl-buttonLink_jmSkY wcl-primary_aIST5 wcl-underline_rL72U"]')
        for i in range(3):
            try:
                await load_more_button_locator.wait_for(
                    state='visible',
                    timeout=2000
                )

                await load_more_button_locator.click()
                await page.wait_for_timeout(1500)
                self.logger.info(f'Am apasat pe "Arata mai multe meciuri" a {i}-a oara')
            except Exception as e:
                self.logger.warning(f'Butonul nu mai exista la click-ul {i + 1}')
                break
        self.logger.info('Am terminat click-urile')
        return await page.content()

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, meta={
                'playwright': True,
                'playwright_page_methods': [PageMethod(self.load_more)]
            })

    def parse(self, response):
        match_urls = response.xpath('//a[@class="eventRowLink"]/@href').getall()
        for url in match_urls:
            stats_url = url.replace('/?mid=', '/sumar/statistici/0/?mid=')
            yield response.follow(
                stats_url,
                callback=self.parse_match_stats_page,
                meta={
                    'playwright': True,
                    'playwright_page_methods': [
                        PageMethod('wait_for_selector', 'xpath=//div[contains(@class, "wcl-category_Ydwqh")]', timeout=10000)]
                }
            )

    def parse_match_stats_page(self, response):
        item = MatchItem()
        item['link_meci'] = response.url
        item['echipa_gazda'] = response.xpath(
            '//div[contains(@class, "duelParticipant__home ")]/div[@class="participant__participantNameWrapper"]/div[@class="participant__participantName participant__overflow"]/a[@class="participant__participantName participant__overflow "]/text()').get()
        item['echipa_oaspete'] = response.xpath(
            '//div[contains(@class, "duelParticipant__away ")]/div[@class="participant__participantNameWrapper"]/div[@class="participant__participantName participant__overflow"]/a[@class="participant__participantName participant__overflow "]/text()').get()
        item['goluri_gazda'] = response.xpath('//div[@class="detailScore__wrapper"]/span[1]/text()').get()
        item['goluri_oaspete'] = response.xpath('//div[@class="detailScore__wrapper"]/span[3]/text()').get()
        item['data_meci'] = response.xpath('//div[@class="duelParticipant__startTime"]/div/text()').get().split(' ')[0]

        stat_rows = response.xpath('//div[contains(@class, "wcl-category_Ydwqh")]')
        self.logger.info(
            f'Procesam statistici pentru meciul: {item['echipa_gazda']} vs {item['echipa_oaspete']}. URL: {item['link_meci']}')
        for row in stat_rows:
            red_card_category = False
            category_name = row.xpath('.//div[2]/strong/text()').get().strip()
            home_val = row.xpath('.//div[1]/strong/text()').get().strip()
            away_val = row.xpath('.//div[3]/strong/text()').get().strip()
            if category_name == 'Posesie minge':
                item['posesie_minge_gazda'] = home_val
                item['posesie_minge_oaspete'] = away_val
            elif category_name == 'Total șuturi':
                item['total_suturi_gazda'] = home_val
                item['total_suturi_oaspete'] = away_val
            elif category_name == 'Șuturi pe poartă':
                item['suturi_pe_poarta_gazda'] = home_val
                item['suturi_pe_poarta_oaspete'] = away_val
            elif category_name == 'Cornere':
                item['cornere_gazda'] = home_val
                item['cornere_oaspete'] = away_val
            elif category_name == 'Cartonașe galbene':
                item['cartonase_galbene_gazda'] = home_val
                item['cartonase_galbene_oaspete'] = away_val
            elif category_name == 'Cartonașe roșii':
                item['cartonase_rosii_gazda'] = home_val
                item['cartonase_rosii_oaspete'] = away_val
                red_card_category = True
            elif category_name == 'Ofsaiduri':
                item['ofsaiduri_gazda'] = home_val
                item['ofsaiduri_oaspete'] = away_val
            elif category_name == 'Lovituri libere':
                item['lovituri_libere_gazda'] = home_val
                item['lovituri_libere_oaspete'] = away_val
            elif category_name == 'Aruncări de la margine':
                item['aruncari_de_la_margine_gazda'] = home_val
                item['aruncari_de_la_margine_oaspete'] = away_val
            elif category_name == 'Intervenții portar':
                item['interventii_portar_gazda'] = home_val
                item['interventii_portar_oaspete'] = away_val
            elif category_name == 'Faulturi':
                item['faulturi_gazda'] = home_val
                item['faulturi_oaspete'] = away_val
            if not red_card_category:
                item['cartonase_rosii_gazda'] = '0'
                item['cartonase_rosii_oaspete'] = '0'
        yield item