#!/usr/bin/env python3
"""
FINKI Course Multimodal Content Scraper
Scrapes course content from FINKI Moodle system with authentication
Downloads PDFs, presentations, and other materials in original format
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin, urlparse
from pathlib import Path
import json
from typing import List, Dict, Any
import mimetypes

class FINKICourseScraper:
    def __init__(self, output_dir: str = "./rag_system/macedonian_data/finki_courses"):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # FINKI course URLs
        self.course_urls = [
            "https://oldcourses.finki.ukim.mk/course/view.php?id=2241",
            "https://oldcourses.finki.ukim.mk/course/view.php?id=2127", 
            "https://oldcourses.finki.ukim.mk/course/view.php?id=2414",
            "https://oldcourses.finki.ukim.mk/course/view.php?id=2232",
            "https://oldcourses.finki.ukim.mk/course/view.php?id=2709"
        ]
        
        # Authentication credentials
        self.username = "223001"
        self.password = "lj29sn04ne16@SK"
        
        # Create output directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track downloaded files
        self.downloaded_files = []
        
        # Supported file extensions for multimodal content
        self.multimodal_extensions = {
            '.pdf', '.ppt', '.pptx', '.doc', '.docx', '.xls', '.xlsx',
            '.zip', '.rar', '.7z', '.mp4', '.avi', '.mkv', '.mp3', 
            '.wav', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg'
        }
        
    def login_to_finki(self) -> bool:
        """Login to FINKI Moodle system via CAS authentication"""
        try:
            print("🔐 Logging into FINKI CAS system...")
            
            # Step 1: Get the CAS login page
            cas_login_url = "https://cas.finki.ukim.mk/cas/login?service=https%3A%2F%2Foldcourses.finki.ukim.mk%2Flogin%2Findex.php"
            response = self.session.get(cas_login_url)
            response.raise_for_status()
            
            print("📄 Fetched CAS login page")
            
            # Step 2: Parse CAS login form
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for the login form - CAS typically uses different form IDs
            login_form = (soup.find('form', {'id': 'fm1'}) or 
                         soup.find('form', {'name': 'f'}) or
                         soup.find('form', attrs={'method': 'post'}) or
                         soup.find('form'))
            
            if not login_form:
                print("❌ Could not find CAS login form")
                print("Available forms:")
                for form in soup.find_all('form'):
                    print(f"  Form: {form.get('id', 'no-id')} {form.get('name', 'no-name')} {form.get('action', 'no-action')}")
                return False
            
            print(f"✅ Found login form: {login_form.get('action', 'no-action')}")
            
            # Step 3: Extract form data and hidden fields
            form_data = {}
            for input_tag in login_form.find_all('input'):
                name = input_tag.get('name')
                value = input_tag.get('value', '')
                input_type = input_tag.get('type', 'text')
                
                if name:
                    form_data[name] = value
                    if input_type == 'hidden':
                        print(f"  Hidden field: {name} = {value}")
            
            # Step 4: Set credentials (CAS typically uses 'username' and 'password')
            form_data['username'] = self.username
            form_data['password'] = self.password
            
            # Sometimes CAS uses different field names
            if 'j_username' in [input_tag.get('name') for input_tag in login_form.find_all('input')]:
                form_data['j_username'] = self.username
                form_data['j_password'] = self.password
            
            print(f"📝 Submitting credentials for user: {self.username}")
            
            # Step 5: Submit to CAS login
            cas_action = login_form.get('action', '')
            if cas_action.startswith('/'):
                cas_submit_url = f"https://cas.finki.ukim.mk{cas_action}"
            elif cas_action.startswith('http'):
                cas_submit_url = cas_action
            else:
                cas_submit_url = f"https://cas.finki.ukim.mk/cas/{cas_action}"
            
            response = self.session.post(cas_submit_url, data=form_data, allow_redirects=True)
            response.raise_for_status()
            
            print(f"🔄 CAS response received, checking authentication...")
            
            # Step 6: Check if we were redirected back to Moodle with authentication
            final_url = response.url
            print(f"  Final URL: {final_url}")
            
            # Check if we're now in the Moodle system
            if "oldcourses.finki.ukim.mk" in final_url:
                # Check for logout link or other indicators of successful login
                if ("logout" in response.text.lower() or 
                    "одјави се" in response.text.lower() or
                    "dashboard" in response.text.lower() or
                    "my courses" in response.text.lower() or
                    "мои курсеви" in response.text.lower()):
                    print("✅ Successfully authenticated via CAS and logged into FINKI")
                    return True
            
            # If we're still on CAS page, check for error messages
            if "cas.finki.ukim.mk" in final_url:
                error_msgs = soup.find_all(['div', 'span'], class_=lambda x: x and 'error' in x.lower())
                if error_msgs:
                    for msg in error_msgs:
                        print(f"❌ CAS Error: {msg.get_text().strip()}")
                else:
                    print("❌ Still on CAS page - authentication may have failed")
            
            print("❌ Login failed - check credentials or CAS configuration")
            return False
                
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def extract_course_info(self, course_url: str) -> Dict[str, Any]:
        """Extract course information and content links"""
        try:
            print(f"📚 Analyzing course: {course_url}")
            
            response = self.session.get(course_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract course title
            title_elem = soup.find('h1') or soup.find('h2') or soup.find('.page-header-headings h1')
            course_title = "Unknown Course"
            if title_elem:
                course_title = title_elem.get_text().strip()
            
            # Extract course ID from URL
            course_id = re.search(r'id=(\d+)', course_url)
            course_id = course_id.group(1) if course_id else "unknown"
            
            print(f"  📖 Course: {course_title}")
            
            # Find all downloadable content links
            content_links = []
            
            # Look for file links
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(course_url, href)
                
                # Check if it's a downloadable file
                if self._is_downloadable_content(full_url, link):
                    # Extract file info
                    file_info = {
                        'url': full_url,
                        'text': link.get_text().strip(),
                        'title': link.get('title', ''),
                        'type': self._guess_content_type(full_url, link)
                    }
                    content_links.append(file_info)
            
            # Look for embedded resources
            for resource in soup.find_all(['img', 'video', 'audio', 'iframe']):
                src = resource.get('src')
                if src:
                    full_url = urljoin(course_url, src)
                    if self._is_downloadable_content(full_url, resource):
                        file_info = {
                            'url': full_url,
                            'text': resource.get('alt', resource.get('title', '')),
                            'title': resource.get('title', ''),
                            'type': self._guess_content_type(full_url, resource)
                        }
                        content_links.append(file_info)
            
            return {
                'course_id': course_id,
                'title': course_title,
                'url': course_url,
                'content_links': content_links
            }
            
        except Exception as e:
            print(f"❌ Error analyzing course {course_url}: {e}")
            return None
    
    def _is_downloadable_content(self, url: str, element) -> bool:
        """Check if URL points to downloadable content"""
        # Check file extension
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Check for file extensions
        for ext in self.multimodal_extensions:
            if path.endswith(ext):
                return True
        
        # Check for Moodle resource/file URLs
        if any(keyword in url.lower() for keyword in ['mod/resource', 'mod/folder', 'pluginfile.php', 'file.php']):
            return True
        
        # Check element attributes
        if hasattr(element, 'get'):
            # Check for download attributes
            if element.get('download') is not None:
                return True
            
            # Check class names that suggest downloadable content
            classes = element.get('class', [])
            if isinstance(classes, list):
                classes = ' '.join(classes)
            if any(keyword in classes.lower() for keyword in ['resource', 'file', 'download', 'attachment']):
                return True
        
        return False
    
    def _guess_content_type(self, url: str, element) -> str:
        """Guess the type of content based on URL and element"""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Check file extension
        for ext in ['.pdf', '.doc', '.docx']:
            if path.endswith(ext):
                return 'document'
        
        for ext in ['.ppt', '.pptx']:
            if path.endswith(ext):
                return 'presentation'
        
        for ext in ['.xls', '.xlsx', '.csv']:
            if path.endswith(ext):
                return 'spreadsheet'
        
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg']:
            if path.endswith(ext):
                return 'image'
        
        for ext in ['.mp4', '.avi', '.mkv', '.mov']:
            if path.endswith(ext):
                return 'video'
        
        for ext in ['.mp3', '.wav', '.ogg']:
            if path.endswith(ext):
                return 'audio'
        
        for ext in ['.zip', '.rar', '.7z']:
            if path.endswith(ext):
                return 'archive'
        
        return 'unknown'
    
    def download_file(self, file_info: Dict[str, Any], course_id: str, course_title: str) -> bool:
        """Download a file maintaining original format"""
        try:
            url = file_info['url']
            print(f"📥 Downloading: {file_info['text'][:50]}...")
            
            # Create course-specific directory
            course_dir = os.path.join(self.output_dir, f"course_{course_id}")
            os.makedirs(course_dir, exist_ok=True)
            
            # Download file
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine filename
            filename = self._get_filename_from_response(response, file_info)
            if not filename:
                # Generate filename from URL or content
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                if not filename or '.' not in filename:
                    # Use content type to generate extension
                    content_type = response.headers.get('content-type', '')
                    ext = mimetypes.guess_extension(content_type) or '.bin'
                    filename = f"file_{len(self.downloaded_files)}{ext}"
            
            # Clean filename
            filename = self._clean_filename(filename)
            filepath = os.path.join(course_dir, filename)
            
            # Avoid overwriting - add number suffix if needed
            counter = 1
            original_filepath = filepath
            while os.path.exists(filepath):
                name, ext = os.path.splitext(original_filepath)
                filepath = f"{name}_{counter}{ext}"
                counter += 1
            
            # Save file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Save metadata
            metadata = {
                'original_url': url,
                'course_id': course_id,
                'course_title': course_title,
                'file_title': file_info.get('text', ''),
                'file_type': file_info.get('type', 'unknown'),
                'filename': filename,
                'filepath': filepath,
                'file_size': len(response.content),
                'content_type': response.headers.get('content-type', 'unknown')
            }
            
            # Save metadata file
            metadata_file = filepath + '.metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.downloaded_files.append(metadata)
            print(f"✅ Saved: {filename} ({len(response.content)} bytes)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {file_info.get('text', 'file')}: {e}")
            return False
    
    def _get_filename_from_response(self, response, file_info: Dict[str, Any]) -> str:
        """Extract filename from response headers or content"""
        # Try Content-Disposition header
        content_disposition = response.headers.get('content-disposition', '')
        if content_disposition:
            filename_match = re.search(r'filename[*]?=["\']?([^"\';\r\n]+)', content_disposition)
            if filename_match:
                return filename_match.group(1)
        
        # Try to use file info text as filename
        text = file_info.get('text', '').strip()
        if text and len(text) < 100:  # Reasonable filename length
            # Clean and add extension if missing
            clean_text = re.sub(r'[<>:"/\\|?*]', '_', text)
            if '.' not in clean_text:
                # Try to guess extension from content type
                content_type = response.headers.get('content-type', '')
                ext = mimetypes.guess_extension(content_type)
                if ext:
                    clean_text += ext
            return clean_text
        
        return None
    
    def _clean_filename(self, filename: str) -> str:
        """Clean filename for filesystem compatibility"""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove extra spaces and dots
        filename = re.sub(r'\s+', '_', filename)
        filename = re.sub(r'\.+', '.', filename)
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        
        return filename
    
    def scrape_all_courses(self):
        """Scrape all FINKI courses"""
        print("🎓 Starting FINKI course scraping...")
        
        # Login first
        if not self.login_to_finki():
            print("❌ Cannot proceed without authentication")
            return
        
        total_files = 0
        
        for course_url in self.course_urls:
            print(f"\n📚 Processing course: {course_url}")
            
            # Extract course info
            course_info = self.extract_course_info(course_url)
            
            if not course_info:
                continue
            
            print(f"  Found {len(course_info['content_links'])} downloadable items")
            
            # Download each file
            for file_info in course_info['content_links']:
                success = self.download_file(
                    file_info, 
                    course_info['course_id'], 
                    course_info['title']
                )
                if success:
                    total_files += 1
                
                # Be respectful - add delay
                time.sleep(1)
        
        # Save overall metadata
        overall_metadata = {
            'scrape_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_courses': len(self.course_urls),
            'total_files_downloaded': total_files,
            'courses_scraped': self.course_urls,
            'downloaded_files': self.downloaded_files
        }
        
        metadata_file = os.path.join(self.output_dir, 'finki_scraping_metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(overall_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 FINKI scraping completed!")
        print(f"📊 Total files downloaded: {total_files}")
        print(f"💾 Files saved to: {self.output_dir}")
        print(f"📋 Metadata saved to: {metadata_file}")

def main():
    """Main function to run the FINKI scraper"""
    scraper = FINKICourseScraper()
    scraper.scrape_all_courses()

if __name__ == "__main__":
    main()
