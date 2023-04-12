"""•The brownie points are calculated at the end of the week and each one gets a goodie package, based on their score.
• 5 brownie points for exercising 1 to 3 hrs
• 10 brownie points for exercising 4 to 6 hrs
• Anything more than 6hrs, you get 15 brownie points.
• Can you help Mr.Beaver write a Python program that calculates the brownie points earned by each member in a week.
• Mr.Beaver wants a list that displays the name of the family member, hours worked and brownie points earned."""
name = (input("Enter child's name "))
hours = int(input("Enter how many hours did they exercise "))
if (hours == 0):
    print(name,"earns 0 points")
elif (hours >= 1) and (hours <= 3):
    print(name,"earns 5 brownie points!")
elif (hours >= 4) and (hours <= 6):
    print(name,"earns 10 brownie points1")
else:
    print(name,"earns 15 points!")