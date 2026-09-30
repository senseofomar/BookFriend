def navigate_matches(matches, index_number, command):
    """
 1. input is matches : list[str] and index_number: int and command: str and output new_index: int and new_match: str
2. if command ==n we increase the index by +1
3. if command ==p we decrease the index by -1
4 if command == number, then index ==int(number)
5. new_match = match[index_number]
6. return (index_number, new_match)
    """

    if command == "n":
        index_number +=1
    elif command == "p":
        index_number -= 1
    elif command.isdigit():
        index_number = int(command)

    else:
        print("Invalid command")
    # clamp index to stay in range
    index_number = max(0, min(index_number, len(matches) - 1))

    new_match = matches[index_number]

    return index_number, new_match
def main():
    matches = ["apple", "poona", "goodf", "hdfdfd"]
    index = 0
    while True:
        command = input("\nEnter your navigation command n, p or index number : ").strip().lower()
        if command == "q":
            print("Exiting..")
            break

        index, new_matches = navigate_matches(matches,index, command)
        print(f"current index: {index}, Match: {new_matches}")
if __name__ =="__main__":
    main()