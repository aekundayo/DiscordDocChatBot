#!/bin/bash

# Set the website URL
website_url="$1"

# Download the website using wget
wget --recursive --no-clobber --page-requisites --html-extension --convert-links --restrict-file-names=windows --no-parent "$website_url"

# Extract domain name from the URL
foldername=$(echo "$website_url" | awk -F[/:] '{print $4}')

# Zip up the downloaded website
zip -r "$foldername.zip" "$foldername/"

