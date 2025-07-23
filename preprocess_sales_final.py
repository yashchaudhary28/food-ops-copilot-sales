#!/usr/bin/env python3
"""
Final Sales Data Preprocessing Script - LLM/SQL Optimized
Restructures table for better LLM understanding and SQL querying
"""

import csv
import re
from datetime import datetime

def clean_value(value):
    """Clean and standardize text values."""
    if not value or value == "-":
        return ""
    
    # Convert to string and lowercase
    cleaned = str(value).lower().strip()
    
    # Replace spaces and special characters with underscores
    cleaned = re.sub(r'[^\w]', '_', cleaned)
    
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    return cleaned

def parse_location(location_str):
    """Parse location string to extract brand and outlet location with robust pattern handling."""
    if not location_str or location_str == "-":
        return "", ""
    
    # Clean the location string first
    location_clean = location_str.strip().strip('"')  # Remove quotes
    
    # Pattern 1: Brand (Outlet Location) - most common
    match = re.match(r'^(.+?)\s*\((.+?)\)$', location_clean)
    if match:
        brand = match.group(1).strip()
        outlet_location = match.group(2).strip()
        return brand, outlet_location
    
    # Pattern 2: Known Brand patterns (PRIORITY - check before dash splitting)
    # Handle cases like "Dhaba Estd 1986 Bhopal", "Dhaba Estd 1986 Delhi - Rohtak"
    known_brands = [
        'Dhaba Estd 1986',
        'Mamagoto', 
        'TBH by SlyGranny', 
        'Outpost'
    ]
    
    for brand in known_brands:
        if location_clean.startswith(brand):
            remaining = location_clean[len(brand):].strip()
            if remaining:
                # Handle cases like "Dhaba Estd 1986 Delhi - Rohtak" -> brand="Dhaba Estd 1986", location="Delhi - Rohtak"
                return brand, remaining
            else:
                return brand, ""  # Brand only, no location
    
    # Pattern 3: Brand - Outlet Location (dash separator)
    if ' - ' in location_clean:
        parts = location_clean.split(' - ', 1)
        brand = parts[0].strip()
        outlet_location = parts[1].strip()
        return brand, outlet_location
    
    # Pattern 4: BRAND LOCATION (all caps, space separated)
    # Handle cases like "MAMAGOTO M3M GGN"
    if location_clean.isupper():
        # Split and take first part as brand, rest as location
        parts = location_clean.split()
        if len(parts) >= 2:
            brand = parts[0]  # First word is brand
            outlet_location = ' '.join(parts[1:])  # Rest is location
            return brand, outlet_location
    
    # Pattern 5: Handle edge cases and fallbacks
    # This should rarely be reached if patterns above work correctly
    
    # Fallback: treat entire string as brand if no clear pattern
    return location_clean, ""

def extract_city_from_location(outlet_location):
    """Extract city from outlet location using specific mappings and patterns."""
    if not outlet_location:
        return ""
    
    # Specific location-to-city mappings (using cleaned values)
    location_city_map = {
        'm3m_ggn': 'gurgaon',
        'world_mark_sector_65': 'delhi',  # aerocity delhi
        'delhi_vasant_kunj': 'delhi',
        'dlf_avenue_mall': 'gurgaon',
        'elan_gurgaon': 'gurgaon',
        'delhi_rohtak': 'delhi',
        'rajouri_garden': 'delhi',
        'noida_sector_62': 'noida',
        'noida_sector_18': 'noida',
        'gurgaon_sector_29': 'gurgaon',
        'gurgaon_mg_road': 'gurgaon'
    }
    
    # Check specific mappings first
    if outlet_location in location_city_map:
        return location_city_map[outlet_location]
    
    # Pattern-based extraction
    if 'delhi' in outlet_location:
        return 'delhi'
    elif 'gurgaon' in outlet_location or 'ggn' in outlet_location:
        return 'gurgaon'
    elif 'noida' in outlet_location:
        return 'noida'
    elif 'mumbai' in outlet_location:
        return 'mumbai'
    elif 'bangalore' in outlet_location or 'bengaluru' in outlet_location:
        return 'bangalore'
    elif 'pune' in outlet_location:
        return 'pune'
    elif 'hyderabad' in outlet_location:
        return 'hyderabad'
    elif 'chennai' in outlet_location:
        return 'chennai'
    elif 'kolkata' in outlet_location:
        return 'kolkata'
    
    # If no pattern matches, return the outlet location as city
    return outlet_location

def standardize_payment_vendor(vendor):
    """Standardize payment vendor names with enhanced cleaning and business focus."""
    if not vendor:
        return ""
    
    vendor_lower = vendor.lower()
    
    # Remove 'other_' prefix and clean unnecessary company names
    if vendor_lower.startswith('other_'):
        vendor_lower = vendor_lower[6:]  # Remove 'other_' prefix
    
    # Standardize online platform variations
    if 'zomato' in vendor_lower:
        return 'zomato'
    elif 'swiggy' in vendor_lower:
        return 'swiggy'
    elif any(variant in vendor_lower for variant in ['eazy_diner', 'eazy_dinner', 'easy_dinner', 'easy_diner']):
        return 'eazy_diner'
    elif 'dineout' in vendor_lower or 'dine_out' in vendor_lower:
        return 'dineout'
    elif 'uber_eats' in vendor_lower or 'ubereats' in vendor_lower:
        return 'uber_eats'
    
    # Standardize UPI variations (clean 'other_' prefix)
    elif 'upi' in vendor_lower:
        return 'upi'
    elif 'paytm' in vendor_lower and 'wallet' not in vendor_lower:
        return 'paytm'
    elif 'phonepe' in vendor_lower or 'phone_pe' in vendor_lower:
        return 'phonepe'
    elif 'gpay' in vendor_lower or 'google_pay' in vendor_lower:
        return 'gpay'
    elif 'bhim' in vendor_lower:
        return 'bhim'
    
    # Standardize card variations
    elif 'amex' in vendor_lower or 'american_express' in vendor_lower:
        return 'amex'
    elif 'visa' in vendor_lower:
        return 'visa'
    elif 'mastercard' in vendor_lower or 'master_card' in vendor_lower:
        return 'mastercard'
    
    # Clean up company names and unnecessary details
    # Remove common business suffixes and person names
    business_suffixes = ['pvt_ltd', 'pvt', 'ltd', 'llp', 'vkllp', 'hospitality', 'ventures', 'kitchen', 'mr_', 'ms_']
    for suffix in business_suffixes:
        if suffix in vendor_lower:
            return 'business'  # Generic business category
    
    # Handle specific known cases
    if vendor_lower in ['cash', 'card', 'digital', 'n/a', 'unknown']:
        return vendor_lower
    
    # For anything else, return a cleaned generic category
    return 'digital'

def parse_payment_info(settlement_mode, payment_gateway, cash_amt, card_amt, amex_amt, other_amt):
    """Parse payment information with enhanced categorization while removing redundant columns."""
    payment_type = ""
    payment_vendor = ""
    
    try:
        # Convert amounts to float, handle None/empty values
        cash_amt = float(cash_amt) if cash_amt and cash_amt != "-" else 0.0
        card_amt = float(card_amt) if card_amt and card_amt != "-" else 0.0
        amex_amt = float(amex_amt) if amex_amt and amex_amt != "-" else 0.0
        other_amt = float(other_amt) if other_amt and other_amt != "-" else 0.0
        
        total_paid = cash_amt + card_amt + amex_amt + other_amt
        
        # Handle null/empty settlement mode and payment gateway based on business context
        settlement_clean = clean_value(settlement_mode) if settlement_mode and settlement_mode != "-" else ""
        gateway_clean = clean_value(payment_gateway) if payment_gateway and payment_gateway != "-" else ""
        
        # If total paid is 0, this is likely a full discount case
        if total_paid == 0:
            payment_type = "Full Discount"
            payment_vendor = "N/A"
        # Enhanced categorization logic (keep the good logic, just don't store settlement/gateway columns)
        else:
            # Check for online payment platforms first (highest priority)
            online_platforms = ['zomato', 'swiggy', 'uber_eats', 'eazy_diner', 'dineout']
            if any(platform in settlement_clean for platform in online_platforms) or \
               any(platform in gateway_clean for platform in online_platforms):
                payment_type = "Online"
                # Determine vendor from either settlement or gateway
                vendor_source = settlement_clean if settlement_clean else gateway_clean
                payment_vendor = standardize_payment_vendor(vendor_source)
            
            # Check for UPI payments
            elif 'upi' in settlement_clean or 'upi' in gateway_clean or \
                 any(upi_app in gateway_clean for upi_app in ['paytm', 'phonepe', 'gpay', 'bhim']):
                payment_type = "UPI"
                payment_vendor = standardize_payment_vendor(gateway_clean) if gateway_clean else "upi"
            
            # Check for wallet payments
            elif 'wallet' in settlement_clean or \
                 any(wallet in gateway_clean for wallet in ['paytm_wallet', 'mobikwik', 'freecharge']):
                payment_type = "Wallet"
                payment_vendor = standardize_payment_vendor(gateway_clean) if gateway_clean else "wallet"
            
            # Check for cash payments
            elif cash_amt > 0 or 'cash' in settlement_clean:
                payment_type = "Cash"
                payment_vendor = "Cash"
            
            # Check for card payments (including Amex)
            elif card_amt > 0 or amex_amt > 0 or \
                 any(card_type in settlement_clean for card_type in ['card', 'credit', 'debit', 'visa', 'mastercard']):
                payment_type = "Card"
                if amex_amt > 0 or 'amex' in gateway_clean:
                    payment_vendor = "Amex"
                else:
                    payment_vendor = standardize_payment_vendor(gateway_clean) if gateway_clean else "Card"
            
            # Fallback: infer from amounts if no clear categorization
            elif not settlement_clean and not gateway_clean:
                if cash_amt > 0:
                    payment_type = "Cash"
                    payment_vendor = "Cash"
                elif card_amt > 0:
                    payment_type = "Card"
                    payment_vendor = "Card"
                elif amex_amt > 0:
                    payment_type = "Card"
                    payment_vendor = "Amex"
                elif other_amt > 0:
                    payment_type = "Digital"
                    payment_vendor = "Digital"
            
            # Final fallback for unclear cases
            else:
                payment_type = "Digital"
                payment_vendor = standardize_payment_vendor(gateway_clean) if gateway_clean else "Digital"
    
    except (ValueError, TypeError):
        payment_type = "Unknown"
        payment_vendor = "Unknown"
    
    return payment_type, payment_vendor

def convert_date(date_str):
    """Convert date string to YYYY-MM-DD format."""
    try:
        # Parse DD/MM/YY format
        dt = datetime.strptime(date_str, '%d/%m/%y')
        return dt.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return None

def format_currency_value(value):
    """Format currency values properly for BigQuery."""
    try:
        if not value or value == "-":
            return 0.0
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def format_integer_value(value):
    """Format integer values properly for BigQuery."""
    try:
        if not value or value == "-":
            return 0
        return int(float(value))  # Convert via float first to handle decimal strings
    except (ValueError, TypeError):
        return 0

def main():
    print("Starting final sales data preprocessing for LLM/SQL optimization...")
    
    # Define column structure optimized for LLM understanding and SQL querying
    # Order: Brand info first, then transaction details, then payment info, then tax details
    final_columns = [
        # Brand and Location Info (First for LLM context)
        'Brand', 'Outlet_Location', 'City', 'Full_Brand_Location',
        
        # Transaction Details
        'Sale_Date', 'Bill_No', 'Channel_Type', 'Pax',
        
        # Financial Core
        'Basic_Amount', 'Discount', 'Net_Sale', 'Settle_Total',
        
        # Payment Information
        'Payment_Type', 'Payment_Vendor', 'Cash', 'Card', 'Amex', 'Other',
        
        # Tax Details (for restaurant tax analysis)
        'CESS_2_Percent', 'CGST_2_5_Percent', 'SGST_2_5_Percent', 
        'Staff_Welfare_Fund_5_Percent', 'Staff_Welfare_Fund_10_Percent',
        'VAT_6_Percent', 'VAT_12_5_Percent', 'VAT_18_9_Percent', 'VAT_25_Percent',
        'Packing_Charge', 'Round_Off'
    ]
    
    # Read and process data
    with open('sale data.csv', 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        # Process rows
        processed_rows = []
        for row in reader:
            # Parse location
            brand, outlet_location = parse_location(row['Location'])
            
            # Parse payment info (use settlement/gateway for categorization but don't store them)
            payment_type, payment_vendor = parse_payment_info(
                row['Settlement Mode'],
                row['Payment Gateway'],
                row['Cash'],
                row['Card'],
                row['Amex'],
                row['Other']
            )
            
            # Convert date
            sale_date = convert_date(row['Date'])
            
            # Clean outlet location first, then extract city
            cleaned_outlet_location = clean_value(outlet_location) if outlet_location else ""
            city = extract_city_from_location(cleaned_outlet_location)
            
            # Create new row with optimized structure
            new_row = {
                # Brand and Location Info (First for LLM context)
                'Brand': clean_value(brand) if brand else "",
                'Outlet_Location': cleaned_outlet_location,
                'City': city or "",
                'Full_Brand_Location': row['Location'] if row['Location'] else "",
                
                # Transaction Details
                'Sale_Date': sale_date or "",
                'Bill_No': row['Bill No.'] if row['Bill No.'] else "",
                'Channel_Type': clean_value(row['Channel Type']) if row['Channel Type'] else "",
                'Pax': format_integer_value(row['Pax']),
                
                # Financial Core
                'Basic_Amount': format_currency_value(row['Basic Amount']),
                'Discount': format_currency_value(row['Discount']),
                'Net_Sale': format_currency_value(row['Net Sale']),
                'Settle_Total': format_currency_value(row['Settle Total']),
                
                # Payment Information
                'Payment_Type': payment_type or "",
                'Payment_Vendor': payment_vendor or "",
                'Cash': format_currency_value(row['Cash']),
                'Card': format_currency_value(row['Card']),
                'Amex': format_currency_value(row['Amex']),
                'Other': format_currency_value(row['Other']),
                
                # Tax Details (for restaurant tax analysis)
                'CESS_2_Percent': format_currency_value(row['CESS 2%']),
                'CGST_2_5_Percent': format_currency_value(row['CGST 2.5%']),
                'SGST_2_5_Percent': format_currency_value(row['SGST 2.5%']),
                'Staff_Welfare_Fund_5_Percent': format_currency_value(row['Staff Welfare Fund 5%']),
                'Staff_Welfare_Fund_10_Percent': format_currency_value(row['Staff Welfare Fund 10%']),
                'VAT_6_Percent': format_currency_value(row['VAT 6%']),
                'VAT_12_5_Percent': format_currency_value(row['VAT 12.5%']),
                'VAT_18_9_Percent': format_currency_value(row['VAT 18.9%']),
                'VAT_25_Percent': format_currency_value(row['VAT 25%']),
                'Packing_Charge': format_currency_value(row['Packing Charge']),
                'Round_Off': format_currency_value(row['Round Off'])
            }
            
            processed_rows.append(new_row)
    
    # Write cleaned data
    with open('cleaned_sales_final.csv', 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=final_columns)
        writer.writeheader()
        writer.writerows(processed_rows)
    
    print(f"Final preprocessing complete! Processed {len(processed_rows)} rows.")
    print("Cleaned data saved to cleaned_sales_final.csv")
    print("\nFinal optimizations:")
    print("- Removed Settlement_Mode and Payment_Gateway columns (redundant)")
    print("- Renamed Location → Full_Brand_Location for LLM clarity")
    print("- Reordered columns: Brand info first, then transaction, payment, tax")
    print("- Included all tax columns for restaurant tax analysis")
    print("- Proper formatting for all numeric columns")
    print("- Optimized column names for SQL querying")
    print(f"Final structure: {len(final_columns)} columns optimized for LLM/SQL usage")

if __name__ == "__main__":
    main()
