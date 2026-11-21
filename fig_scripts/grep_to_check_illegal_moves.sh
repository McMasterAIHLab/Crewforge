# finds all occurances of the illegal move notification. This will appear many times because it is in agents' history.
grep -r -oE "ILLEGAL MOVE ALERT: The proposed move '[^']*'" gcc_focused_data/focused_runs/ | uniq -c

# take the moves that above command found, and run. This will show how many times the illegal move was proposed. Run once for each illegal move listed above
grep -F -o '"move": "<move that matched above>"' <file that matched above>

