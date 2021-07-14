import extract_transform
import staging
import load


# execute the whole ETL one step at a time
# could have written a bash script
extract_transform.main()
staging.main()
load.main()

print('ETL process completed.')

