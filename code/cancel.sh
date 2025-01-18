#export myJOBQueue="bfx-jq-metaorf"

for state in SUBMITTED PENDING RUNNABLE STARTING RUNNING
do 
    echo "$state"
    for job in $(aws batch list-jobs --job-queue $myJOBQueue --job-status $state --output text --query "jobSummaryList[*].[jobId]" --region us-west-2)
    do 
        echo "Stopping job $job in state $state"
        aws batch cancel-job --reason "Terminating job." --job-id $job --region us-west-2
    done
done
