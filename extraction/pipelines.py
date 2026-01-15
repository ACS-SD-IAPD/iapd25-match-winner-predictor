import csv


class SaveToCsvPipeline:
    def open_spider(self, spider):
        self.file = open('flashscore_stats.csv', 'w', newline='', encoding='utf-8')
        self.fieldnames = ['link_meci', 'data_meci', 'echipa_gazda', 'echipa_oaspete', 'goluri_gazda', 'goluri_oaspete',
                           'posesie_minge_gazda', 'posesie_minge_oaspete', 'total_suturi_gazda', 'total_suturi_oaspete',
                           'suturi_pe_poarta_gazda', 'suturi_pe_poarta_oaspete', 'cornere_gazda', 'cornere_oaspete',
                           'cartonase_galbene_gazda', 'cartonase_galbene_oaspete', 'cartonase_rosii_gazda',
                           'cartonase_rosii_oaspete',
                           'ofsaiduri_gazda', 'ofsaiduri_oaspete', 'lovituri_libere_gazda', 'lovituri_libere_oaspete',
                           'aruncari_de_la_margine_gazda',
                           'aruncari_de_la_margine_oaspete', 'faulturi_gazda', 'faulturi_oaspete',
                           'interventii_portar_gazda', 'interventii_portar_oaspete']
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        spider.logger.info('Am scris header-ul fisierului csv')

    def close_spider(self, spider):
        self.file.close()
        spider.logger.info('Am scris fisierul csv')

    def process_item(self, item, spider):
        self.writer.writerow(item)
        return item