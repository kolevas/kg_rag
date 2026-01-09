import requests
from bs4 import BeautifulSoup
import os
import time
import random
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set
import re
import json

class MacedonianWebScraper:
    def __init__(self, output_dir: str = "./macedonian_data"):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Macedonian news and content websites
        self.macedonian_sources = [
            "https://www.mkd.mk",
            "https://www.slobodenpecat.mk", 
            "https://www.akademik.mk",
            "https://www.time.mk",
            "https://www.nova.mk",
            "https://www.ukim.edu.mk",
            "https://www.fil.ukim.edu.mk",
            "https://mk.wikipedia.org",
            "https://www.24tv.mk",
            "https://www.telma.com.mk",
            "https://www.kanal5.com.mk",
            "https://www.alfa.mk",
            "https://www.sitel.com.mk",
            "https://www.24info.mk",
            "https://www.kurir.mk",
            "https://www.press24.mk",
            "https://www.dnevnik.mk",
            "https://www.vest.mk",
            "https://www.plusinfo.mk",
            "https://www.republika.mk",
            "https://www.600000.mk",
            "https://www.libertas.mk",
            "https://www.frontline.mk",
            "https://www.360stepeni.mk",
            "https://www.infomax.mk",
            "https://www.makfax.com.mk",
            "https://www.mia.mk",
            "https://www.fom.edu.mk",
            "https://www.finki.ukim.mk",
            "https://mk.voanews.com",
            "https://www.dw.com/mk",
            "https://www.bbc.com/macedonian",
        ]
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track scraped URLs to avoid duplicates
        self.scraped_urls: Set[str] = set()
        
    def is_macedonian_text(self, text: str) -> bool:
        """Check if text contains Macedonian Cyrillic characters"""
        macedonian_chars = re.findall(r'[АБВГДЃЕЖЗЅИЈКЛЉМНЊОПРСТЌУФХЦЧЏШабвгдѓежзѕијклљмнњопрстќуфхцчџш]', text)
        return len(macedonian_chars) > 50  # Threshold for considering text as Macedonian
        
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Macedonian
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-АБВГДЃЕЖЗЅИЈКЛЉМНЊОПРСТЌУФХЦЧЏШабвгдѓежзѕијклљмнњопрстќуфхцчџш]', '', text)
        return text.strip()
        
    def extract_article_content(self, url: str) -> Dict[str, str]:
        """Extract main content from a webpage"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts, styles, and other non-content elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'menu']):
                element.decompose()
            
            # Try to find main content areas
            content_selectors = [
                'article', 'main', '.content', '.post-content', 
                '.entry-content', '.article-content', '.news-content',
                '.text', '.story-content'
            ]
            
            content = ""
            title = ""
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                title = self.clean_text(title_tag.get_text())
            
            # Try different content selectors
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    for element in elements:
                        text = element.get_text()
                        if self.is_macedonian_text(text):
                            content += text + "\n"
                    break
            
            # If no specific content area found, extract from body
            if not content:
                body = soup.find('body')
                if body:
                    # Extract paragraphs
                    paragraphs = body.find_all('p')
                    for p in paragraphs:
                        text = p.get_text()
                        if self.is_macedonian_text(text):
                            content += text + "\n"
            
            content = self.clean_text(content)
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'length': len(content)
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def get_links_from_page(self, base_url: str, max_links: int = 25) -> List[str]:
        """Extract links from a webpage"""
        try:
            response = self.session.get(base_url, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                full_url = urljoin(base_url, href)
                
                # Filter for relevant links (articles, news, etc.)
                if any(keyword in href.lower() for keyword in ['вест', 'статија', 'новост', 'текст', 'post', 'article', 'news', 'vest', 'novost', 'clanak', 'writeup', 'storija', 'izvestaj', 'objaveno', 'objava']):
                    if full_url not in self.scraped_urls:
                        links.append(full_url)
                        
                if len(links) >= max_links:
                    break
            
            # If no specific article links found, try to get recent content links
            if len(links) < 5:
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    full_url = urljoin(base_url, href)
                    
                    # Look for links that might contain content
                    if (full_url.startswith('http') and 
                        base_url in full_url and 
                        full_url != base_url and
                        not any(skip in href.lower() for skip in ['contact', 'about', 'javascript:', 'mailto:', '#', 'login', 'register']) and
                        full_url not in self.scraped_urls):
                        
                        links.append(full_url)
                        
                    if len(links) >= max_links:
                        break
                    
            return links[:max_links]
            
        except Exception as e:
            print(f"Error getting links from {base_url}: {e}")
            return []
    
    def scrape_macedonian_content(self, max_articles: int = 200):
        """Main scraping method"""
        print("🔍 Starting Macedonian content scraping...")
        
        scraped_articles = []
        articles_count = 0
        
        for source in self.macedonian_sources:
            if articles_count >= max_articles:
                break
                
            print(f"\n📰 Scraping from: {source}")
            
            # Get article links from the main page
            links = self.get_links_from_page(source, max_links=25)
            print(f"Found {len(links)} potential article links")
            
            for link in links:
                if articles_count >= max_articles:
                    break
                    
                if link in self.scraped_urls:
                    continue
                    
                print(f"📄 Scraping article: {link}")
                
                article_data = self.extract_article_content(link)
                
                if article_data and article_data['content'] and len(article_data['content']) > 100:
                    self.scraped_urls.add(link)
                    scraped_articles.append(article_data)
                    articles_count += 1
                    
                    # Save individual article
                    filename = f"macedonian_article_{articles_count:03d}.txt"
                    filepath = os.path.join(self.output_dir, filename)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Title: {article_data['title']}\n")
                        f.write(f"URL: {article_data['url']}\n")
                        f.write(f"Length: {article_data['length']} characters\n")
                        f.write("-" * 80 + "\n")
                        f.write(article_data['content'])
                    
                    print(f"✅ Saved: {filename} ({article_data['length']} chars)")
                else:
                    print("❌ Skipped: No Macedonian content or too short")
                
                # Random delay to be respectful
                time.sleep(random.uniform(1, 3))
        
        # Save metadata
        metadata_file = os.path.join(self.output_dir, "scraping_metadata.json")
        metadata = {
            'total_articles': len(scraped_articles),
            'sources_scraped': self.macedonian_sources,
            'scrape_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'articles_summary': [
                {
                    'title': article['title'][:100],
                    'url': article['url'],
                    'length': article['length']
                } for article in scraped_articles
            ]
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 Scraping completed!")
        print(f"📊 Total articles scraped: {len(scraped_articles)}")
        print(f"💾 Files saved to: {self.output_dir}")
        print(f"📋 Metadata saved to: {metadata_file}")
        
        return scraped_articles

def main():
    """Example usage"""
    scraper = MacedonianWebScraper()
    
    # Scrape Macedonian content with more articles
    articles = scraper.scrape_macedonian_content(max_articles=150)
    
    if articles:
        print(f"\nSample scraped content:")
        print(f"Title: {articles[0]['title']}")
        print(f"Content preview: {articles[0]['content'][:200]}...")

if __name__ == "__main__":
    main()
