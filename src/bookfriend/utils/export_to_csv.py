import csv


def export_to_csv(results, filename ):
    if not results:
        print("⚠️ Saved nothing to recent_search_results.")
        return

    with open(filename, 'w', encoding = 'utf-8', newline = '') as f :
        writer = csv.writer(f)
        writer.writerow(["Chapter", "Sentence", "Snippet"])
        writer.writerows(results)
    print("✅ your recent searches are saved in ", filename)

    cmd = input("\nDo you want a custom csv file for your search keywords? ( type y or yes **anything apart from that means NO**): ").strip().lower()
    if cmd in ("yes", "y"):
        custom_filename = (input("Enter your csv file name (with.csv) : ")).strip()
        with open(custom_filename, 'w', encoding= 'utf-8', newline= '') as f:
            writer = csv.writer(f)
            writer.writerow(["Chapter", "Sentence", "Snippet"])
            writer.writerows(results)
        print(f"✅ Your custom CSV file is saved as {custom_filename}")



