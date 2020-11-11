#!/usr/bin/perl -w

$data_points = shift;
$attribute_n = shift;
$value_n = shift;

$n=0;
while ($n<$data_points) {
    $original_data_point = 0;
    while (!($original_data_point)) {
	@att_values = ();
	$curr_n = 0;
	while ($curr_n<$attribute_n) {
	    push @att_values,int(rand($value_n));
	    $curr_n++;
	}
	$data_point_as_string = join " ",@att_values;
	if (!$seen_data_points{$data_point_as_string}) {
	    $seen_data_points{$data_point_as_string}=1;
	    print join(" ",@att_values),"\n";
	    $original_data_point = 1;
	}
    }
    $n++;
}
