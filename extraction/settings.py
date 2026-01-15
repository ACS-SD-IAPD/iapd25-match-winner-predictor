BOT_NAME = 'extraction'

SPIDER_MODULES = [f'{BOT_NAME}.spiders']
NEWSPIDER_MODULE = f'{BOT_NAME}.spiders'

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
ROBOTSTXT_OBEY = False

DOWNLOAD_DELAY = 2
CONCURRENT_REQUESTS_PER_DOMAIN = 4

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0


DOWNLOAD_HANDLERS = {
    "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
    "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
}
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT = 30000

ITEM_PIPELINES = {
   f'{BOT_NAME}.pipelines.SaveToCsvPipeline': 300,
}

LOG_LEVEL = 'INFO'

RETRY_TIMES = 3

DOWNLOAD_TIMEOUT = 60